import torch
import torch.nn

import distdl

from .layers import Concatenate
from distdl_unet import UNetBase
from distdl_unet import UNetLevelBase

_layer_type_map = {
    "conv": (None, distdl.nn.DistributedConv1d, distdl.nn.DistributedConv2d, distdl.nn.DistributedConv3d),
    "pool": (None, distdl.nn.DistributedMaxPool1d, distdl.nn.DistributedMaxPool2d, distdl.nn.DistributedMaxPool3d)
}

class DistributedUNet(UNetBase):

    def __init__(self, P, *args, **kwargs):

        self.P = P
        self.feature_dimension = len(P.shape[2:])
        self.ConvType = _layer_type_map["conv"][feature_dimension]
        self.PoolType = _layer_type_map["pool"][feature_dimension]

        super(DistrubutedUNet, self).__init__(*args, **kwargs)


    def assemble_input_map(self):

        conv = self.ConvType(self.P,
                             in_channels=self.in_channels,
                             out_channels=self.base_channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=self.base_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(conv, norm, acti)

    def assemble_unet(self):
        return DistributedUNetLevel(self.P,
                                    self.levels, 0, 0, self.base_channels, **self.level_kwargs)

    def assemble_output_map(self):

        conv =self.ConvType(self.P,
                            in_channels=self.base_channels,
                            out_channels=self.out_channels,
                            kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = distdl.nn.DistributedBatchNorm(self.P,
        #                                       num_features=self.out_channels)
        # acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(conv)  #, norm, acti)


class DistributedUNetLevel(UNetLevelBase):

    def __init__(self, P, *args, **kwargs):

        self.P = P
        self.feature_dimension = len(P.shape[2:])
        self.ConvType = _layer_type_map["conv"][feature_dimension]
        self.PoolType = _layer_type_map["pool"][feature_dimension]

        super(DistributedUNetLevel, self).__init__(*args, **kwargs)

    def _smoothing_block(self):

        channels = self.channels()
        conv = self.ConvType(self.P,
                             in_channels=channels,
                             out_channels=channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=channels)
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

        pool = self.PoolType(self.P, kernel_size=2, stride=2)
        conv = self.ConvType(self.P,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(pool, conv, norm, acti)

    def assemble_prolongation(self):

        in_channels = self.channels(self.level)
        out_channels = self.channels(self.level-1)

        up = distdl.nn.DistributedUpsample(self.P,
                                           scale_factor=2)
        conv = self.ConvType(self.P,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = distdl.nn.DistributedBatchNorm(self.P,
        #                                       num_features=out_channels)
        # acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(up, conv)  #, norm, acti)

    def assemble_correction(self):

        in_channels = 2*self.channels()
        out_channels = self.channels()

        add = Concatenate(1)
        conv = self.ConvType(self.P,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(add, conv, norm, acti)

    def instantiate_sublevel(self, order):
        return DistributedUNetLevel(self.P, self.max_levels, self.level+1, order, self.base_channels,
                                    nu_1=self.nu_1, nu_2=self.nu_2, nu_e=self.nu_e, mu=self.mu,
                                    reuse_sublevels=self.reuse_sublevels)