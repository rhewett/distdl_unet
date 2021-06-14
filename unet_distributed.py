import torch
import torch.nn

import distdl

from layers import Concatenate
from unet_base import UNetBase
from unet_base import UNetLevelBase

class DistributedUNet(UNetBase):

    def __init__(self, P, *args, **kwargs):
        super(DistrubutedUNet, self).__init__(*args, **kwargs)

        self.P = P

    def assemble_input_map(self):

        conv = distdl.nn.DistributedConv3d(self.P,
                                           in_channels=self.in_channels,
                                           out_channels=self.base_channels,
                                           kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=self.out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(conv, norm, acti)

    def assemble_unet(self):
        return DistributedUNetLevel(self.P,
                                    self.levels, 0, 0, self.base_channels, **self.level_kwargs)

    def assemble_output_map(self):

        conv =distdl.nn.DistributedConv3d(self.P,
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
        super(DistributedUNetLevel, self).__init__(*args, **kwargs)

        self.P = P

    def _smoothing_block(self):

        channels = self.channels()
        conv = distdl.nn.DistributedConv3d(self.P,
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

        pool = distdl.nn.MaxPool3d(self.P, kernel_size=2, stride=2)
        conv = distdl.nn.DistributedConv3d(self.P,
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
        conv = distdl.nn.DistributedConv3d(self.P,
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
        conv = distdl.nn.DistributedConv3d(self.P,
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