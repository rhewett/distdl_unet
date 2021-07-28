from memory_profiler import profile

profiler_outfile_base = "memprof_seq"

fp=open(f'{profiler_outfile_base}.log','w+')

@profile(stream=fp)
def target():

    import os, os.path
    import sys

    import numpy as np
    import torch

    from distdl_unet import ClassicalUNet
    from distdl_unet.logging_timer import LoggingTimer

    from random_ellipses import gen_data
    from eval import iou


    outfile = sys.argv[1]

    timer = LoggingTimer()

    input_features = tuple([int(v) for v in sys.argv[2:]])
    feature_dimension = len(input_features)

    #################################

    depth = 3
    in_channels = 1
    base_channels = 64
    out_channels = 1

    nu_1 = 1
    nu_2 = 1
    nu_e = 1
    mu = 1

    unet = ClassicalUNet(feature_dimension, depth, in_channels, base_channels, out_channels,
                         nu_1=nu_1, nu_2=nu_2, nu_e=nu_e, mu=mu)

    #################################

    n_batch = 1
    batch_size = 1

    sample_spacing = [np.linspace(0, 1, f) for f in input_features]
    sample_grid = np.meshgrid(*sample_spacing)

    n_ellipses_target = 3
    n_ellipses_noise = 2

    timer.start("data gen")
    batches = list()
    for i in range(n_batch):
        batch = list()
        for j in range(batch_size):
            # Add an image-mask tuple to the batch
            batch.append(gen_data(sample_grid, n_ellipses_target, n_ellipses_noise))
        img = torch.cat([im for im, ma in batch],dim=0)
        mask = torch.cat([ma for im, ma in batch],dim=0)

        batches.append((img, mask))
    timer.stop("data gen", input_features)

    ################################

    n_epoch = 10

    parameters = [p for p in unet.parameters()]
    optimizer = torch.optim.Adam(parameters,lr=0.0001)

    criterion = torch.nn.BCEWithLogitsLoss()

    #################################

    # We define the inner loop separately so we get a separate profile log
    @profile
    def train_inner(j, i):

        timer.start("batch")
        img, mask = batches[i]

        optimizer.zero_grad()

        timer.start("forward")
        out = unet(img)
        timer.stop("forward", f"{j}, {i}")

        timer.start("loss")
        loss = criterion(out, mask)
        loss_value = loss.item()
        iou_value = iou(out>0.5, mask>0)
        print(f"Loss: {loss_value}\tIOU: {iou_value}")
        timer.stop("loss", f"{j}, {i}")

        timer.start("adjoint")
        loss.backward()
        timer.stop("adjoint", f"{j}, {i}")

        timer.start("step")
        optimizer.step()
        timer.stop("step", f"{j}, {i}")

        timer.stop("batch", f"{j}, {i}")

        timer.to_csv(outfile)


    for j in range(n_epoch):

        timer.start("epoch")

        for i in range(n_batch):

            train_inner(j, i)

        timer.stop("epoch", j)

    timer.to_csv(outfile)


from mprof_wrapper import mprof_wrap
mprof_wrap(target, outfile_base=profiler_outfile_base)
fp.close()