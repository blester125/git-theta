#!/usr/bin/env python3

import argparse
import operator as op

import numpy as np
import torch
from git_theta_merge_cli import utils
from git_theta_merge_cli.merges import reductions

from git_theta import git_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model-1")
parser.add_argument("--model-2")

args = parser.parse_args()

model_1 = torch.load(args.model_1)
model_2 = torch.load(args.model_2)


def flatten(model):
    return torch.hstack(
        [p.view(-1) for n, p in sorted(model.items(), key=op.itemgetter(0))]
    )


dot = torch.dot(flatten(model_1), flatten(model_2))

repo = git_utils.get_git_repo()
m1 = utils.load_metadata(repo, args.model_1).flatten()
m2 = utils.load_metadata(repo, args.model_2).flatten()
d = reductions.Dot()

our_dot = d.run_merge([m1, m2], [], {})

np.testing.assert_allclose(dot.numpy(), our_dot.numpy(), rtol=1e-5)
