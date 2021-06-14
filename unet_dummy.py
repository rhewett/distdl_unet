import torch
import torch.nn

from layers import Concatenate
from unet_base import UNetBase
from unet_base import UNetLevelBase

class DummyLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, level, action,
                       *args, **kwargs):
        super(DummyLayer, self).__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.action = action

    def __repr__(self):

        ci = self.in_channels
        co = self.out_channels

        k = 2**self.level

        if self.action == "S":
            fi = f"n/{k}"
            fo = f"n/{k}"
        elif self.action == "R":
            fi = f"n/{k}"
            fo = f"n/{k*2}"
        elif self.action == "P":
            fi = f"n/{k*2}"
            fo = f"n/{k}"

        return f"DummyLayer({fi} x {ci} -> {fo} x {co})"

class DummyUNet(UNetBase):

    def __init__(self, *args, **kwargs):
        super(DummyUNet, self).__init__(*args, **kwargs)

    def assemble_input_map(self):

        return DummyLayer(self.in_channels, self.base_channels, 0, "S")

    def assemble_unet(self):
        return DummyUNetLevel(self.levels, 0, 0, self.base_channels, **self.level_kwargs)

    def assemble_output_map(self):
        return DummyLayer(self.base_channels, self.in_channels, 0, "S")


class DummyUNetLevel(UNetLevelBase):

    def __init__(self, *args, **kwargs):
        super(DummyUNetLevel, self).__init__(*args, **kwargs)

    def _smoothing_block(self):

        channels = self.channels()
        return DummyLayer(channels, channels, self.level, "S")

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
        return DummyLayer(in_channels, out_channels, self.level, "R")


    def assemble_prolongation(self):

        in_channels = self.channels(self.level+1)
        out_channels = self.channels(self.level)
        return DummyLayer(in_channels, out_channels, self.level, "P")

    def assemble_correction(self):

        in_channels = 2*self.channels()
        out_channels = self.channels()
        return DummyLayer(in_channels, out_channels, self.level, "S")

    def instantiate_sublevel(self, order):
        return DummyUNetLevel(self.max_levels, self.level+1, order, self.base_channels,
                              nu_1=self.nu_1, nu_2=self.nu_2, nu_e=self.nu_e, mu=self.mu,
                              reuse_sublevels=self.reuse_sublevels)