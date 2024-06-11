#!/usr/bin/env python3

import argpare
import torch

paresr = argparse.ArgumentParser(desciption="Flip all the signs in a model.")
parser.add_argument("--model", required=True, help="The model to filp.")
args = parser.parse_args()


m = torch.load(args.model)
# Pytorch models are flat when loaded.
m = {k: v * -1 for k, v in m.items()}
torch.save(m, args.model)
