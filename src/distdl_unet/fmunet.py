import torch
import torch.nn

from .layers import Concatenate
from distdl_unet import MuNetBase
from distdl_unet import MuNetLevelBase

from .unet_classic import ClassicalUNet
from .unet_classic import ClassicalUNetLevel

_layer_type_map = {
    "conv": (None, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d),
    "pool": (None, torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d),
    "norm": (None, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
}

class InnerVCycle(ClassicalUNet):

    def init(self, *args, **kwargs):

        super(InnerVCycle, self).__init__(*args, **kwargs)

    def assemble_input_map(self):
        return torch.nn.Identity()

    def assemble_output_map(self):
        return torch.nn.Identity()

# This can be any old fmunet
class FMuNet(ClassicalUNet):

    def __init__(self, *args,inner_munet_args=None, **kwargs):

        if inner_munet_args is None:
            self.inner_munet_args = {}
        else:
            self.inner_munet_args = inner_munet_args

        super(FMuNet, self).__init__(*args, **kwargs)

    def assemble_cycle(self):
        return FMuNetLevel(self.feature_dimension,
                           self.levels, 0, 0, self.base_channels,
                           **self.level_kwargs)


class FMuNetLevel(ClassicalUNetLevel):

    def assemble_inner_munet(self):
        depth = self.max_levels - (self.level)
        in_channels = self.base_channels
        base_channels = self.base_channels
        out_channels = self.base_channels
        return InnerVCycle(self.feature_dimension,
                           depth, in_channels, base_channels, out_channels,
                           nu_1=1, nu_2=1, nu_e=1, mu=1,
                           reuse_sublevels=self.reuse_sublevels)

    def assemble_post_smooth(self):
        if self.level == 0:
            blocks = list()
            for k in range(self.nu_2):
                blocks.append(self._smoothing_block())
            return torch.nn.Sequential(*blocks)
        else:
            return self.assemble_inner_munet()

    def instantiate_sublevel(self, order):
        return FMuNetLevel(self.feature_dimension,
                           self.max_levels, self.level+1, order, self.base_channels,
                           nu_1=self.nu_1, nu_2=self.nu_2, nu_e=self.nu_e, mu=self.mu,
                           reuse_sublevels=self.reuse_sublevels)

    def assemble_coarsest_smooth(self):
        depth = self.max_levels - (self.level+1)
        print(depth)
        assert 0
        in_channels = self.base_channels
        base_channels = self.base_channels
        out_channels = self.base_channels
        return InnerVCycle(self.feature_dimension,
                           depth, in_channels, base_channels, out_channels,
                           nu_1=1, nu_2=1, nu_e=1, mu=1,
                           reuse_sublevels=self.reuse_sublevels)
