import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class REFUGEDataset(CustomDataset):

    def __init__(self, **kwargs):
        super(REFUGEDataset, self).__init__(
            **kwargs)
        assert osp.exists(self.img_dir)