# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from ..builder import MODELS, build_module
from ..common import set_requires_grad
from .base_gan import BaseGAN

# _SUPPORT_METHODS_ = ['DCGAN', 'STYLEGANv2']


# @MODELS.register_module(_SUPPORT_METHODS_)
@MODELS.register_module()
class MyGAN(BaseGAN):

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 auxiliary_discriminator=None,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self._gen_cfg = deepcopy(generator)
        self.generator = build_module(generator)

        # support no discriminator in testing
        if discriminator is not None:
            self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None

        if auxiliary_discriminator is not None:
            self.auxiliary_discriminator = build_module(auxiliary_discriminator)
        else:
            self.auxiliary_discriminator = None

            # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_module(gan_loss)
        else:
            self.gan_loss = None

        if disc_auxiliary_loss:
            self.disc_auxiliary_losses = build_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        self.lamda = self.train_cfg.get('lamda', 0.01)
        self.lamda_aux = self.train_cfg.get('lamda_aux', 0.002)
        self.source = self.train_cfg.get('source', 'refuge')

        # self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def _get_aux_disc_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_aux_target'] = 0.5 * self.gan_loss(
            outputs_dict['disc_aux_pred_target'], outputs_dict['label_target'])
        losses_dict['loss_disc_aux_source'] = 0.5 * self.gan_loss(
            outputs_dict['disc_aux_pred_source'], outputs_dict['label_source'])

        # disc auxiliary loss
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

    def _get_disc_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_target'] = 0.5 * self.gan_loss(
            outputs_dict['disc_pred_target'], outputs_dict['label_target'])
        losses_dict['loss_disc_source'] = 0.5 * self.gan_loss(
            outputs_dict['disc_pred_source'], outputs_dict['label_source'])

        # disc auxiliary loss
        if self.with_disc_auxiliary_loss:
            for loss_module in self.disc_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

    def _get_gen_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_gen_target'] = self.lamda * self.gan_loss(
            outputs_dict['disc_pred_target'], outputs_dict['label_target'])
        if outputs_dict['with_auxiliary_head']:
            losses_dict['loss_gen_target_aux'] = self.lamda_aux * self.gan_loss(
                outputs_dict['disc_aux_pred_target'], outputs_dict['label_target'])

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

    def train_step(self,
                   data,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            loss_scaler (:obj:`torch.cuda.amp.GradScaler` | None, optional):
                The loss/gradient scaler used for auto mixed-precision
                training. Defaults to ``None``.
            use_apex_amp (bool, optional). Whether to use apex.amp. Defaults to
                ``False``.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        # get data from data_batch
        # real_imgs = data_batch[self.real_img_key]
        img_s, gt_seg, img_metas_s = data[0]['img'], data[0]['gt_semantic_seg'], data[0]['img_metas']
        img_t, img_metas_t = data[1]['img'], data[1]['img_metas']
        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = img_s.shape[0]
        B_s = img_s.shape[0]
        B_t = img_t.shape[0]
        label_s = torch.ones((B_s, 1, 512, 512), device=img_s.device)
        label_t = torch.zeros((B_t, 1, 512, 512), device=img_t.device)
        label_adv = torch.ones((B_t, 1, 512, 512), device=img_t.device)

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        with_auxiliary_head = self.auxiliary_discriminator!=None

        # generator training
        set_requires_grad(self.discriminator, False)
        if with_auxiliary_head:
            set_requires_grad(self.auxiliary_discriminator, False)
        optimizer['generator'].zero_grad()

        _, target_seg = self.generator(img_t, img_metas_t, return_loss=False, with_auxiliary_head=with_auxiliary_head, source=self.source)
        disc_auxi_pred_target_g = None
        if with_auxiliary_head:
            img_num = target_seg.shape[0] // 2
            auxiliary_target_seg = target_seg[img_num:]
            target_seg = target_seg[:img_num]
            disc_auxi_pred_target_g = self.auxiliary_discriminator(auxiliary_target_seg)
        disc_pred_target_g = self.discriminator(target_seg)
        # todo
        _, source_seg = self.generator(img_s, img_metas_s, return_loss=True, with_auxiliary_head=with_auxiliary_head, source=self.source)
        auxiliary_source_seg = None
        if with_auxiliary_head:
            img_num = source_seg.shape[0] // 2
            auxiliary_source_seg = source_seg[img_num:]
            source_seg = source_seg[:img_num]

        data_dict_ = dict(
            disc_pred_target=disc_pred_target_g,
            disc_aux_pred_target=disc_auxi_pred_target_g,
            label_target=label_adv,
            aux_pred=source_seg,
            aux_source_seg=auxiliary_source_seg,
            aux_label=gt_seg,
            with_auxiliary_head=with_auxiliary_head,
            source=self.source
        )

        loss_gen, log_vars_g = self._get_gen_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        if loss_scaler:
            loss_scaler.scale(loss_gen).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_gen, optimizer['generator'],
                    loss_id=1) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_gen.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['generator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['generator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['generator'].step()

        # disc training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        if with_auxiliary_head:
            set_requires_grad(self.auxiliary_discriminator, True)
            optimizer['auxiliary_discriminator'].zero_grad()

        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            # fake_imgs = self.generator(None, num_batches=batch_size)
            _, pred_s = self.generator(img_s, img_metas_s, return_loss=False, with_auxiliary_head=with_auxiliary_head, source=self.source)
            _, pred_t = self.generator(img_t, img_metas_t, return_loss=False, with_auxiliary_head=with_auxiliary_head, source=self.source)

        disc_aux_pred_t = None
        disc_aux_pred_s = None
        if with_auxiliary_head:
            n1 = pred_s.shape[0] // 2
            aux_pred_s = pred_s[n1:]
            pred_s = pred_s[:n1]
            n2 = pred_t.shape[0] // 2
            aux_pred_t = pred_t[n2:]
            pred_t = pred_t[:n2]
            disc_aux_pred_t = self.auxiliary_discriminator(aux_pred_t)
            disc_aux_pred_s = self.auxiliary_discriminator(aux_pred_s)
        # disc pred for fake imgs and real_imgs
        disc_pred_t = self.discriminator(pred_t)
        disc_pred_s = self.discriminator(pred_s)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            disc_pred_target=disc_pred_t,
            disc_pred_source=disc_pred_s,
            disc_aux_pred_target=disc_aux_pred_t,
            disc_aux_pred_source=disc_aux_pred_s,
            label_target=label_t,
            label_source=label_s,
            with_auxiliary_head=with_auxiliary_head)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_disc, optimizer['discriminator'],
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['discriminator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['discriminator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['discriminator'].step()

        if with_auxiliary_head:
            loss_aux_disc, log_vars_disc = self._get_aux_disc_loss(data_dict_)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_aux_disc))
            if loss_scaler:
                # add support for fp16
                loss_scaler.scale(loss_aux_disc).backward()
            elif use_apex_amp:
                from apex import amp
                with amp.scale_loss(
                        loss_aux_disc, optimizer['auxiliary_discriminator'],
                        loss_id=0) as scaled_aux_loss_disc:
                    scaled_aux_loss_disc.backward()
            else:
                loss_aux_disc.backward()

            if loss_scaler:
                loss_scaler.unscale_(optimizer['auxiliary_discriminator'])
                # note that we do not contain clip_grad procedure
                loss_scaler.step(optimizer['auxiliary_discriminator'])
                # loss_scaler.update will be called in runner.train()
            else:
                optimizer['auxiliary_discriminator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)

        results = dict(source_seg=source_seg.cpu(), target_seg=target_seg.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def forward(self, img, img_metas, return_loss=False, **kwargs):
        output = self.generator.forward_test(img, img_metas, **kwargs)
        return output
