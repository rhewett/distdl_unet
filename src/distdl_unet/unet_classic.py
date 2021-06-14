import torch
import torch.nn

from .layers import Concatenate
from distdl_unet import UNetBase
from distdl_unet import UNetLevelBase

_layer_type_map = {
    "conv": (None, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d),
    "pool": (None, torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d),
    "norm": (None, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
}

class ClassicalUNet(UNetBase):

    def __init__(self, feature_dimension, *args, **kwargs):

        self.feature_dimension = feature_dimension
        self.ConvType = _layer_type_map["conv"][feature_dimension]
        self.PoolType = _layer_type_map["pool"][feature_dimension]
        self.NormType = _layer_type_map["norm"][feature_dimension]

        super(ClassicalUNet, self).__init__(*args, **kwargs)

    def assemble_input_map(self):

        conv = self.ConvType(in_channels=self.in_channels,
                             out_channels=self.base_channels,
                             kernel_size=3, padding=1)
        norm = self.NormType(num_features=self.base_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(conv, norm, acti)

    def assemble_unet(self):
        return ClassicalUNetLevel(self.feature_dimension,
                                  self.levels, 0, 0, self.base_channels,
                                  **self.level_kwargs)

    def assemble_output_map(self):

        conv = self.ConvType(in_channels=self.base_channels,
                             out_channels=self.out_channels,
                             kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = self.NormType(num_features=self.out_channels)
        # acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(conv)  #, norm, acti)


class ClassicalUNetLevel(UNetLevelBase):

    def __init__(self, feature_dimension, *args, **kwargs):

        self.feature_dimension = feature_dimension
        self.ConvType = _layer_type_map["conv"][feature_dimension]
        self.PoolType = _layer_type_map["pool"][feature_dimension]
        self.NormType = _layer_type_map["norm"][feature_dimension]

        super(ClassicalUNetLevel, self).__init__(*args, **kwargs)

    def _smoothing_block(self):

        channels = self.channels()
        conv = self.ConvType(in_channels=channels,
                             out_channels=channels,
                             kernel_size=3, padding=1)
        norm = self.NormType(num_features=channels)
        acti = torch.nn.ReLU(inplace=True)

        return torch.nn.Sequential(conv, norm, acti)

    def assemble_coarsest_smooth(self):

        blocks = list()
        for k in range(self.nu_e):
            blocks.append(self._smoothing_block())
        return torch.nn.Sequential(*blocks)

    def assemble_pre_smooth(self):

        blocks = list()
        for k in range(self.nu_1):
            blocks.append(self._smoothing_block())
        return torch.nn.Sequential(*blocks)

    def assemble_post_smooth(self):

        blocks = list()
        for k in range(self.nu_2):
            blocks.append(self._smoothing_block())
        return torch.nn.Sequential(*blocks)

    def assemble_restriction(self):

        in_channels = self.channels()
        out_channels = self.channels(self.level+1)

        pool = self.PoolType(kernel_size=2, stride=2)
        conv = self.ConvType(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3, padding=1)
        norm = self.NormType(num_features=out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(pool, conv, norm, acti)

    def assemble_prolongation(self):

        in_channels = self.channels(self.level+1)
        out_channels = self.channels(self.level)

        up = torch.nn.Upsample(scale_factor=2)
        conv = self.ConvType(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = self.NormType(num_features=out_channels)
        # acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(up, conv)  #, norm, acti)

    def assemble_correction(self):

        in_channels = 2*self.channels()
        out_channels = self.channels()

        add = Concatenate(1)
        conv = self.ConvType(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3, padding=1)
        norm = self.NormType(num_features=out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(add, conv, norm, acti)

    def instantiate_sublevel(self, order):
        return ClassicalUNetLevel(self.feature_dimension,
                                  self.max_levels, self.level+1, order, self.base_channels,
                                  nu_1=self.nu_1, nu_2=self.nu_2, nu_e=self.nu_e, mu=self.mu,
                                  reuse_sublevels=self.reuse_sublevels)