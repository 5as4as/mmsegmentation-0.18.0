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
class MultiTaskGAN(BaseGAN):

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 auxiliary_discriminator=None,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 kd_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self._gen_cfg = deepcopy(generator)
        self.generator = build_module(generator)

        # support no discriminator in testing
        if discriminator is not None:
            if isinstance(discriminator, list):
                self.discriminator = nn.ModuleList()
                for head_cfg in discriminator:
                    self.discriminator.append(build_module(head_cfg))
            else:
                self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None

        if auxiliary_discriminator is not None:
            if isinstance(auxiliary_discriminator, list):
                self.auxiliary_discriminator = nn.ModuleList()
                for head_cfg in auxiliary_discriminator:
                    self.auxiliary_discriminator.append(build_module(head_cfg))
            else:
                self.auxiliary_discriminator = build_module(auxiliary_discriminator)
        else:
            self.auxiliary_discriminator = None

            # support no gan_loss in testing
        # todo, support multi discriminator with different gan_loss
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

        if kd_loss:
            if isinstance(kd_loss, list):
                self.kd_loss = nn.ModuleList()
                for item in kd_loss:
                    self.kd_loss.append(build_module(item))
            else:
                self.kd_loss = build_module(kd_loss)
        else:
            self.kd_loss = None

        if self.kd_loss is not None:
            self.with_kd = True
        else:
            self.with_kd = False

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

        self.num_task = 2

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

        # self.ODOC_lamda = self.train_cfg.get('ODOC_lamda', 0.01)
        # self.ODOC_lamda_aux = self.train_cfg.get('ODOC_lamda_aux', 0.002)
        # # self.source = self.train_cfg.get('source', 'refuge')
        #
        # self.lesion_lamda = self.train_cfg.get('lesion_lamda', 0.01)
        # self.lesion_lamda_aux = self.train_cfg.get('lesion_lamda_aux', 0.002)

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
        disc_aux_pred = outputs_dict['disc_aux_pred']
        for i in range(len(disc_aux_pred)):
            for j in range(len(disc_aux_pred[i])):
                if j == 0:
                    losses_dict[f'loss_disc_aux_source_{i}'] = 0.5 * self.gan_loss(
                        disc_aux_pred[i][j], outputs_dict['label'][0])
                else:
                    losses_dict[f'loss_disc_aux_target_{i}_{j}'] = 0.5 * self.gan_loss(
                        disc_aux_pred[i][j], outputs_dict['label'][1])
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
        disc_pred = outputs_dict['disc_pred']
        for i in range(len(disc_pred)):
            for j in range(len(disc_pred[i])):
                if j == 0:
                    losses_dict[f'loss_disc_source_{i}'] = 0.5 * self.gan_loss(
                        disc_pred[i][j], outputs_dict['label'][0])
                else:
                    losses_dict[f'loss_disc_target_{i}_{j}'] = 0.5 * self.gan_loss(
                        disc_pred[i][j], outputs_dict['label'][1])

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
        num_task = len(outputs_dict['aux_pred'])
        per_lamda = isinstance(self.lamda, list)
        for i in range(num_task):
            for j in range(len(outputs_dict['disc_pred_target'][i])):
                if per_lamda:
                    losses_dict[f'loss_gen_target_{i}_{j}'] = self.lamda[i] * self.gan_loss(
                        outputs_dict['disc_pred_target'][i][j], outputs_dict['label_target'])
                else:
                    losses_dict[f'loss_gen_target_{i}_{j}'] = self.lamda * self.gan_loss(
                        outputs_dict['disc_pred_target'][i][j], outputs_dict['label_target'])
        if outputs_dict['with_auxiliary_head']:
            for i in range(num_task):
                for j in range(len(outputs_dict['disc_aux_pred_target'][i])):
                    if per_lamda:
                        losses_dict[f'loss_gen_target_aux_{i}_{j}'] = self.lamda[i] * self.gan_loss(
                            outputs_dict['disc_aux_pred_target'][i][j], outputs_dict['label_target'])
                    else:
                        losses_dict[f'loss_gen_target_{i}_{j}'] = self.lamda * self.gan_loss(
                            outputs_dict['disc_pred_target'][i][j], outputs_dict['label_target'])

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                losses_dict.update(loss_)
        # if self.with_kd:
        #     if 'kd_pred1' in outputs_dict:
        #         losses_dict['loss_kd_1'] = self.kd_loss[0](outputs_dict['kd_pred1'], outputs_dict['kd_label1'])  # 视杯视盘预测头
        #         losses_dict['loss_kd_2'] = self.kd_loss[1](outputs_dict['kd_pred2'], outputs_dict['kd_label2'])  # 病灶分割头
        #     else:
        #         losses_dict['loss_kd_2'] = self.kd_loss(outputs_dict['kd_pred2'], outputs_dict['kd_label2'])  # 病灶分割头
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

    def train_step(self,
                   data,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        # get data from data_batch
        # real_imgs = data_batch[self.real_img_key]
        imgs = []
        gt_segs = []
        img_metas = []
        for d in data:
            img, gt_seg, img_meta = d['img'], d['gt_semantic_seg'], d['img_metas']
            imgs.append(img)
            gt_segs.append(gt_seg)
            img_metas.append(img_meta)

        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = imgs[0].shape[0]
        B_1 = imgs[0].shape[0]
        # B_2 = imgs[1].shape[0]
        label_1 = torch.ones((B_1, 1, 512, 512), device=imgs[0].device)
        label_2 = torch.zeros((B_1, 1, 512, 512), device=imgs[0].device)
        label_adv = torch.ones((B_1, 1, 512, 512), device=imgs[0].device)
        # label_adv_2 = torch.ones((B_1, 1, 512, 512), device=imgs[0].device)
        label = [label_1, label_2]
        # label_adv = [label_adv_1, label_adv_2]

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

        # [decoder1_seg, decoder2_seg]
        raw_segs = []
        act_segs = []
        for i in range(len(imgs)):
            raw, act = self.generator(imgs[i], img_metas[i],  with_auxiliary_head=with_auxiliary_head)
            raw_segs.append(raw)
            act_segs.append(act)
        # raw_seg_1, act_seg_1 = self.generator(imgs[0], img_metas[0],  with_auxiliary_head=with_auxiliary_head)
        # raw_seg_2, act_seg_2 = self.generator(imgs[1], img_metas[1],  with_auxiliary_head=with_auxiliary_head)

        # if self.with_kd:
        #     if isinstance(raw_seg_1[0], list):
        #         source_seg = [raw_seg_1[0][0], raw_seg_2[1][0]]
        #     else:
        #         source_seg = [raw_seg_1[0], raw_seg_2[1][0]]
        # else:
        #      source_seg = [raw_seg_1[0], raw_seg_2[1]]
        # target_seg = [act_seg_2[0], act_seg_1[1]]
        source_seg = []
        target_seg = []
        for i in range(len(imgs)):
            source_seg.append(raw_segs[i][i])
            tmp = []
            for j in range(len(imgs)):
                if j != i:
                    tmp.append(act_segs[j][i])
            target_seg.append(tmp)

        disc_pred_target_g = []
        disc_auxi_pred_target_g = []
        if with_auxiliary_head:
            for i, ts in enumerate(target_seg):
                tmp = []
                for j in range(len(ts)):
                    img_num = ts[j].shape[0] // 2
                    auxiliary_target_seg = ts[j][img_num:]
                    target_seg[i][j] = ts[j][:img_num]
                    tmp.append(self.auxiliary_discriminator[i](auxiliary_target_seg))
                disc_auxi_pred_target_g.append(tmp)

        for i, ts in enumerate(target_seg):
            tmp = []
            for j in range(len(ts)):
                tmp.append(self.discriminator[i](ts[j]))
            disc_pred_target_g.append(tmp)

        auxiliary_source_seg = []
        if with_auxiliary_head:
            for i in range(len(source_seg)):
                img_num = source_seg[i].shape[0] // 2
                auxiliary_source_seg.append(source_seg[i][img_num:])
                source_seg[i] = source_seg[i][:img_num]

        data_dict_ = dict(
            disc_pred_target=disc_pred_target_g,
            disc_aux_pred_target=disc_auxi_pred_target_g,
            label_target=label_adv,
            aux_pred=source_seg,
            aux_source_seg=auxiliary_source_seg,
            aux_label=gt_segs,
            with_auxiliary_head=with_auxiliary_head,
            with_kd=self.with_kd
        )
        # if self.with_kd:
        #     if isinstance(raw_seg_1[0], list):    # 视杯视盘预测头的病灶分割
        #         img_num = raw_seg_1[1][0].shape[0] // 2  # 默认蒸馏情况下，病灶分割预测头携带蒸馏头
        #         data_dict_['kd_label1'] = raw_seg_1[1][0][:img_num].detach()
        #         data_dict_['kd_pred1'] = raw_seg_1[0][1]
        #         data_dict_['kd_task1_pred'] = raw_seg_2[0][1]
        #     if isinstance(raw_seg_2[0], list):
        #         img_num = raw_seg_2[0][0].shape[0] // 2
        #         data_dict_['kd_label2'] = raw_seg_2[0][0][:img_num].detach()
        #     else:
        #         img_num = raw_seg_2[0].shape[0] // 2
        #         data_dict_['kd_label2'] = raw_seg_2[0][:img_num].detach()
        #     data_dict_['kd_pred2'] = raw_seg_2[1][1]
        #     data_dict_['kd_task2_pred'] = raw_seg_1[1][1]

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
        for i in range(len(act_segs)):
            for j in range(len(act_segs[i])):
                act_segs[i][j] = act_segs[i][j].detach()
        # for i in range(len(act_seg_1)):
        #     act_seg_1[i] = act_seg_1[i].detach()
        #     act_seg_2[i] = act_seg_2[i].detach()

        preds = []
        for i in range(len(act_segs)):
            tmp = []
            tmp.append(act_segs[i][i])
            for j in range(len(act_segs)):
                if j != i:
                    tmp.append(act_segs[j][i])
            preds.append(tmp)
        # pred_1 = [act_seg_1[0], act_seg_2[0]]
        # pred_2 = [act_seg_2[1], act_seg_1[1]]
        disc_aux_pred = []
        # disc_aux_pred_1 = []
        # disc_aux_pred_2 = []
        if with_auxiliary_head:
            for i in range(len(preds)):
                tmp = []
                for j in range(len(preds[i])):
                    n1 = preds[i][j].shape[0] // 2
                    aux_pred = preds[i][j][n1:]
                    preds[i][j] = preds[i][j][:n1]
                    tmp.append(self.auxiliary_discriminator[i](aux_pred))
                disc_aux_pred.append(tmp)
        # disc pred for fake imgs and real_imgs
        disc_pred = []
        # disc_pred_1 = []
        # disc_pred_2 = []
        for i in range(len(preds)):
            tmp = []
            for j in range(len(preds[i])):
                tmp.append(self.discriminator[i](preds[i][j]))
            disc_pred.append(tmp)

        # get data dict to compute losses for disc
        data_dict_ = dict(
            disc_pred=disc_pred,
            disc_aux_pred=disc_aux_pred,
            label=label,
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
            loss_aux_disc, log_vars_aux_disc = self._get_aux_disc_loss(data_dict_)
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
        if with_auxiliary_head:
            log_vars.update(log_vars_aux_disc)

        outputs = dict(
            log_vars=log_vars, num_samples=batch_size)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def forward(self, img, img_metas, return_loss=False, **kwargs):
        output = self.generator.forward_test(img, img_metas, **kwargs)
        return output
