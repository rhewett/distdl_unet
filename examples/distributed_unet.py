from mpi4py import MPI

import os, os.path
import sys

import numpy as np
import torch

import distdl
from distdl.backends.mpi.partition import MPIPartition

from distdl_unet import DistributedUNet
from distdl_unet.logging_timer import MPILoggingTimer

from random_ellipses import gen_data
from eval import iou

# program_name logfile.csv nf1 nf2 ... nfd pf1 pf2 ... pfd
print(sys.argv)

# Setup logging output
outfile = sys.argv[1]
timer = MPILoggingTimer()


# Parse configuration inputs
input_info = [int(v) for v in sys.argv[2:]]

feature_dimension = len(input_info) // 2
input_workers = tuple(input_info[feature_dimension:])
input_features = tuple(input_info[:feature_dimension])

n_workers = np.prod(input_workers)

# Setup some partitions:
# 1) P_world: all possible workers
# 2) P_base: enough workers to satisfy n_workers
# 3) P_0/P_root: A partition to create data on (until we create it in parallel)
# 4) P_unet: partition given by input_workers of appropriate dimension

P_world = MPIPartition(MPI.COMM_WORLD)
P_base = P_world.create_partition_inclusive(np.arange(n_workers))

# 2+feature_dimension comes batch x channel x f0 x f1 x ... fd
P_root_shape = [1]*(2+feature_dimension)
P_0 = P_base.create_partition_inclusive([0])
P_root = P_0.create_cartesian_topology_partition(P_root_shape)

P_unet_shape = [1, 1] + list(input_workers)
P_unet = P_base.create_cartesian_topology_partition(P_unet_shape)

#################################

depth = 1
in_channels = 1
base_channels = 64
out_channels = 1

nu_1 = 1
nu_2 = 1
nu_e = 1
mu = 1

unet = DistributedUNet(P_root, P_unet,
                       depth, in_channels, base_channels, out_channels,
                       nu_1=nu_1, nu_2=nu_2, nu_e=nu_e, mu=mu)

#################################

n_batch = 1
batch_size = 1

MPI.COMM_WORLD.Barrier()
if P_root.active:

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
else:
    timer.start("data gen")
    batches = list()
    for i in range(n_batch):
        batch = list()
        for j in range(batch_size):
            img = distdl.utilities.torch.zero_volume_tensor(batch_size)
            mask = distdl.utilities.torch.zero_volume_tensor(batch_size)

        batches.append((img, mask))
    timer.stop("data gen", input_features)

MPI.COMM_WORLD.Barrier()

#################################

n_epoch = 1

parameters = [p for p in unet.parameters()]

# Hack to make empty parts of the graph happy
if not parameters:
    parameters = [torch.nn.Parameter(torch.zeros(1))]

optimizer = torch.optim.Adam(parameters,lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

################################

for j in range(n_epoch):

    MPI.COMM_WORLD.Barrier()

    timer.start("epoch")

    for i in range(n_batch):

        MPI.COMM_WORLD.Barrier()

        timer.start("batch")
        img, mask = batches[i]

        optimizer.zero_grad()

        timer.start("forward")
        out = unet(img)
        timer.stop("forward", f"{j}, {i}")

        timer.start("loss")
        if P_root.active:
            loss = criterion(out, mask)
            loss_value = loss.item()
            iou_value = iou(out>0.5, mask>0)
            print(f"Loss: {loss_value}\tIOU: {iou_value}")
        else:
            loss = out.clone()
        timer.stop("loss", f"{j}, {i}")

        timer.start("adjoint")
        loss.backward()
        timer.stop("adjoint", f"{j}, {i}")

        timer.start("step")
        optimizer.step()
        timer.stop("step", f"{j}, {i}")

        timer.stop("batch", f"{j}, {i}")

        if P_root.active:
            timer.to_csv(outfile)

    timer.stop("epoch", j)

if P_root.active:
    timer.to_csv(outfile)
