import torch
import torch.nn

class UNetBase(torch.nn.Module):

    def __init__(self, levels, in_channels, base_channels, out_channels, **level_kwargs):
        super(UNetBase, self).__init__()

        self.levels = levels

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels

        self.level_kwargs = level_kwargs

        self.input_map = self.assemble_input_map()
        self.unet = self.assemble_unet()
        self.output_map = self.assemble_output_map()

    def assemble_input_map(self):
        raise NotImplementedError()

    def assemble_unet(self):
        raise NotImplementedError()

    def assemble_output_map(self):
        raise NotImplementedError()

    def forward(self, input):

        x_f = self.input_map(input)
        y_f = self.unet(x_f)
        output = self.output_map(y_f)


class UNetLevelBase(torch.nn.Module):

    def __init__(self, max_levels, level, order, base_channels,
                 nu_1=1, nu_2=1, nu_e=1, mu=1, reuse_sublevels=False):

        super(UNetLevelBase, self).__init__()

        self.max_levels = max_levels
        self.level = level
        self.order = order

        self.base_channels = base_channels

        # Number of pre-relaxation iterations
        self.nu_1 = nu_1

        # Number of post-relaxation iterations
        self.nu_2 = nu_2

        # Number of "exact" solve iterations
        self.nu_e = nu_e

        # mu factor from Briggs, et al
        self.mu = mu

        self.reuse_sublevels = reuse_sublevels

        self.coarsest = (self.level == self.max_levels-1)

        if self.coarsest:
            self.coarsest_smooth = self.assemble_coarsest_smooth()
        else:
            self.pre_smooth = self.assemble_pre_smooth()
            self.restriction = self.assemble_restriction()
            self.sublevels = self.assemble_sublevels()
            self.prolongation = self.assemble_prolongation()
            self.correction = self.assemble_correction()
            self.post_smooth = self.assemble_post_smooth()

    def channels(self, level=None):
        if level is None:
            level = self.level
        return (2**level)*self.base_channels

    def assemble_coarsest_smooth(self):
        raise NotImplementedError()

    def assemble_pre_smooth(self):
        raise NotImplementedError()

    def assemble_post_smooth(self):
        raise NotImplementedError()

    def assemble_restriction(self):
        raise NotImplementedError()

    def assemble_prolongation(self):
        raise NotImplementedError()

    def assemble_correction(self):
        raise NotImplementedError()

    def instantiate_sublevel(self, order):
        raise NotImplementedError()

    def assemble_sublevels(self):

        # If this level is less than one less than the max, it is a coarsest level
        if not self.coarsest:
            # First sublevel is order 0
            sublevels = [self.instantiate_sublevel(0)]

            if self.level == self.max_levels-2:
                # "Exact solve" is not repeated
                pass
            elif self.reuse_sublevels:
                # Build the sublevel level once then replicate it
                sublevels = sublevels*self.mu
            else:
                # Take that first sublevel and build a new sublevel for each other subcycle
                for i in range(1, self.mu):
                    sublevel = self.instantiate_sublevel(i)
                    sublevels.append(sublevel)
            return torch.nn.Sequential(*sublevels)
        else:
            raise Exception()

    def forward(self, x_f):

        if self.coarsest:
            y_f = self.coarsest_smooth(x_f)
            return y_f

        y_f = self.pre_smooth(x_f)
        y_c = self.restriction(y_f)

        for sublevel in self.sublevel_levels:
            y_c = sublevel(y_c)

        y_c = self.prolongation(y_c)
        y_f = self.correction((y_f, y_c))
        y_f = self.post_smooth(y_f)