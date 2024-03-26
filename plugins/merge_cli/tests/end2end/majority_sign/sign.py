#!/usr/bin/env python3

import argparse
import operator as op

import numpy as np
import torch
from git_theta_merge_cli import utils
from git_theta_merge_cli.merges import reductions

from git_theta import git_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model")

args = parser.parse_args()

model = torch.load(args.model)


def flatten(model):
    return torch.hstack(
        [p.view(-1) for n, p in sorted(model.items(), key=op.itemgetter(0))]
    )


sign = torch.sign(torch.sum(torch.sign(flatten(model))))

repo = git_utils.get_git_repo()
m = utils.load_metadata(repo, args.model).flatten()
s = reductions.Sign()

our_sign = s.run_merge([m], [], {})

np.testing.assert_allclose(sign.numpy(), our_sign.numpy(), rtol=1e-5)
