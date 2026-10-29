# Swamp

A data generator for teaching PCA. A hidden swamp holds a fly, whose brain
is a few neurons with a few genes each, and a food web of five species.
Both evolve as linear oscillators, so the data hide in a few principal
components. `swamp` writes the full state at consecutive ticks as a CSV
matrix.

## Install

With [uv](https://docs.astral.sh/uv/), one command sets up a virtual
environment in `.venv`, installs numpy and scipy, and runs the generator:

```bash
uv run swamp --neurons 3 --genes 2 --seed 42 --samples 1000 > data.csv
```

Without uv, any Python 3.10 or newer will do:

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e .
```

```bash
.venv/bin/swamp --neurons 3 --genes 2 --seed 42 --samples 1000 > data.csv
```

## Usage

Every flag has a default: `--neurons` in the fly's brain 2, `--genes` per
neuron 0, `--seed` 0 and `--samples` 1000. The number of samples can also
be given on its own, as in `swamp 500`. The seed and the two sizes fix the
world. Row t is the state at tick t, for
t = 0, 1, and so on. The header names the columns: the fly's position `x`
and `y`, firing rates `r1` to `rN`, gene levels `g1_1` to `gN_G` (neuron,
then gene), then the population count of each species. With no neurons the
fly has no brain, and `x` and `y` stay at 0.

## The hidden world

The fly block is M_f x'' = -K_f x over firing rates r (N) and gene levels g
(N G). K_f is built from a neuron coupling A, a gene network B shared by all
neurons and a gene-to-rate coupling c:

    K_f = [[A, kron(I, c^T)], [kron(I, c), kron(I, B)]]

Gene masses are 100 times the rate masses, so genetic modes are slow. The
fly starts at rest, with standard normal rates and genes. The fly sits at
(x, y) = P r, with P a random 2 x N matrix of standard normal entries.

The population block is p'' = -K_p p over deviations from equilibrium, with
off-diagonal stiffness only on predator-prey links. The web is fixed: frogs
eat flies, snakes eat frogs, herons eat frogs and snakes, and alligators eat
frogs, herons and snakes. Equilibria lie between 10^2 and 10^3, and
deviations start near a tenth of them, so standardize the columns before
PCA.

All parameters and initial states come from the seed. A draw is accepted only
if both K are positive definite and every population stays positive for all
time. Otherwise the generator draws again from the same random stream. Both
blocks evolve exactly through their normal modes.

## Choices

Every value lives in `swamp/constants.py`.

| constant | value | meaning |
| --- | --- | --- |
| M_R, M_G | 1, 100 | rate and gene masses |
| A_DIAG, A_OFFDIAG_SD | U(0.4, 1.6), 0.15 | neuron coupling A |
| B_DIAG, B_OFFDIAG_SD | U(0.3, 1.0), 0.15 | gene network B |
| C_SD | 0.3 | gene-to-rate coupling c |
| POP_DIAG, POP_LINK | U(1, 2), +-U(0.3, 1) | K_p diagonal and links |
| POP_STIFFNESS | 0.05 | scale of K_p |
| EQUILIBRIUM | log-U(10^2, 10^3) | equilibrium population sizes |
| POP_DEVIATION, POP_VEL_SCALE | 0.1, 0.1 | initial deviations, rates |
| MAX_REDRAWS | 10^4 | draws tried before giving up |
| STATE_DECIMALS | 3 | decimals written to the matrix |
