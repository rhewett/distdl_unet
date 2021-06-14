import torch
import torch.nn

from layers import Concatenate
from unet_base import UNetBase
from unet_base import UNetLevelBase

class ClassicalUNet(UNetBase):

    def __init__(self, *args, **kwargs):
        super(ClassicalUNet, self).__init__(*args, **kwargs)

    def assemble_input_map(self):

        conv = torch.nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.base_channels,
                               kernel_size=3, padding=1)
        norm = torch.nn.BatchNorm3d(num_features=self.out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(conv, norm, acti)

    def assemble_unet(self):
        return ClassicalUNetLevel(self.levels, 0, 0, self.base_channels, **self.level_kwargs)

    def assemble_output_map(self):

        conv = torch.nn.Conv3d(in_channels=self.base_channels,
                               out_channels=self.out_channels,
                               kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = torch.nn.BatchNorm3d(num_features=self.out_channels)
        # acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(conv)  #, norm, acti)


class ClassicalUNetLevel(UNetLevelBase):

    def __init__(self, *args, **kwargs):
        super(ClassicalUNetLevel, self).__init__(*args, **kwargs)

    def _smoothing_block(self):

        channels = self.channels()
        conv = torch.nn.Conv3d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3, padding=1)
        norm = torch.nn.BatchNorm3d(num_features=channels)
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

        pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        conv = torch.nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, padding=1)
        norm = torch.nn.BatchNorm3d(num_features=out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(pool, conv, norm, acti)

    def assemble_prolongation(self):

        in_channels = self.channels(self.level+1)
        out_channels = self.channels(self.level)

        up = torch.nn.Upsample(scale_factor=2)
        conv = torch.nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = torch.nn.BatchNorm3d(num_features=out_channels)
        # acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(up, conv)  #, norm, acti)

    def assemble_correction(self):

        in_channels = 2*self.channels()
        out_channels = self.channels()

        add = Concatenate(1)
        conv = torch.nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, padding=1)
        norm = torch.nn.BatchNorm3d(num_features=out_channels)
        acti = torch.nn.ReLU(inplace=True)
        return torch.nn.Sequential(add, conv, norm, acti)

    def instantiate_sublevel(self, order):
        return ClassicalUNetLevel(self.max_levels, self.level+1, order, self.base_channels,
                                  nu_1=self.nu_1, nu_2=self.nu_2, nu_e=self.nu_e, mu=self.mu,
                                  reuse_sublevels=self.reuse_sublevels)