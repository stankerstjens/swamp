"""The hidden world: a fly and a food web, each obeying M x'' = -K x."""

import numpy as np
from scipy import linalg

from . import constants as C

SPECIES = ["flies", "frogs", "snakes", "herons", "alligators"]
# Row eats column.
EATS = np.array([
    [0, 0, 0, 0, 0],  # flies
    [1, 0, 0, 0, 0],  # frogs
    [0, 1, 0, 0, 0],  # snakes
    [0, 1, 1, 0, 0],  # herons
    [0, 1, 1, 1, 0],  # alligators
])
S = len(SPECIES)


def draw_symmetric(rng, n, diag, offdiag_sd):
    X = np.triu(rng.normal(0, offdiag_sd, (n, n)), 1)
    return X + X.T + np.diag(rng.uniform(*diag, n))


def modes(masses, K, x0, v0):
    """(w, V, a, b) with x(t) = V (a cos wt + b sin wt), or None if K is
    not positive definite."""
    w2, V = linalg.eigh(K, np.diag(masses))
    if np.any(w2 <= 0):
        return None
    w = np.sqrt(w2)
    return w, V, V.T @ (masses * x0), V.T @ (masses * v0) / w


def draw(rng, N, G):
    """Modes and initial states of both blocks, or None if rejected."""
    A = draw_symmetric(rng, N, C.A_DIAG, C.A_OFFDIAG_SD)
    B = draw_symmetric(rng, G, C.B_DIAG, C.B_OFFDIAG_SD)
    c = rng.normal(0, C.C_SD, G)
    P = rng.normal(0, 1, (2, N))
    coupling = np.kron(np.eye(N), c[None, :])
    K_f = np.block([[A, coupling],
                    [coupling.T, np.kron(np.eye(N), B)]])
    masses_f = np.r_[np.full(N, C.M_R), np.full(N * G, C.M_G)]

    signs = rng.choice([-1.0, 1.0], (S, S))
    offdiag = signs * rng.uniform(*C.POP_LINK, (S, S)) * EATS.T
    K_p = C.POP_STIFFNESS * (offdiag + offdiag.T
                             + np.diag(rng.uniform(*C.POP_DIAG, S)))
    eq = np.exp(rng.uniform(*np.log(C.EQUILIBRIUM), S))

    x0 = rng.normal(0, 1, N + N * G)
    dev = C.POP_DEVIATION * eq
    p0 = rng.normal(0, dev)
    v0 = rng.normal(0, dev * C.POP_VEL_SCALE)

    fly = modes(masses_f, K_f, x0, np.zeros_like(x0))
    pop = modes(np.ones(S), K_p, p0, v0)
    if fly is None or pop is None:
        return None
    # The largest swing of each population must stay below its equilibrium.
    w, V, a, b = pop
    if np.any(np.abs(V) @ np.hypot(a, b) >= eq):
        return None
    return fly, P, pop, eq


def evolve(w, V, a, b, t):
    wt = np.outer(t, w)
    return (a * np.cos(wt) + b * np.sin(wt)) @ V.T


def matrix(N, G, seed, samples):
    """Rows are ticks 0, 1, ...; columns are the fly's position, rates,
    genes and populations."""
    rng = np.random.default_rng(seed)
    for _ in range(C.MAX_REDRAWS):
        world = draw(rng, N, G)
        if world is not None:
            break
    else:
        raise RuntimeError("no acceptable draw")
    fly, P, pop, eq = world
    t = np.arange(samples)
    x = evolve(*fly, t)
    return np.hstack([x[:, :N] @ P.T, x, eq + evolve(*pop, t)])
