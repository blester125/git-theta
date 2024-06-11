#!/usr/bin/env python3

import argparse

import torch

parser = argparse.ArgumentParser(description="")
parser.add_argument("--models", nargs="+", help="")


def main(args):
    models = [torch.load(m) for m in args.models]

    summed = {}
    for name in models[0]:
        summed[name] = torch.sum(torch.stack([m[name] for m in models], dim=0), dim=0)
    flat = torch.hstack([v.ravel() for v in summed.values()])
    signs = torch.sign(flat)
    majority = torch.sign(torch.sum(signs))
    majority = majority.masked_fill(majority == 0, 1)
    print(majority)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
