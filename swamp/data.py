"""The full state of one world at consecutive ticks, as a CSV matrix."""

import argparse
import sys

import numpy as np

from . import constants as C
from .model import SPECIES, matrix


def columns(N, G):
    rates = [f"r{i + 1}" for i in range(N)]
    genes = [f"g{i + 1}_{j + 1}" for i in range(N) for j in range(G)]
    return ["x", "y"] + rates + genes + SPECIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=2)
    parser.add_argument("--genes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("count", type=int, nargs="?", metavar="SAMPLES",
                        help="same as --samples")
    args = parser.parse_args()
    N, G = args.neurons, args.genes
    samples = args.samples if args.count is None else args.count
    np.savetxt(sys.stdout, matrix(N, G, args.seed, samples),
               delimiter=",", fmt=f"%.{C.STATE_DECIMALS}f",
               header=",".join(columns(N, G)), comments="")
