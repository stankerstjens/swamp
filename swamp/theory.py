"""Drains from the energy intake of the reference strategies, in theory.

A fly is a sum of normal modes with fixed amplitudes A_m and random phases,
so with a_m = A_m cos(phase) and b_m = A_m sin(phase) every position is
linear in (a, b). Each strategy aims with a linear guess, so its miss is
G (a, b) plus rounding noise for some 2 x 2n matrix G. Mode m then adds a
vector that turns around an ellipse, whose characteristic function is a
Bessel function, and the chance q that the miss lands within the catch radius
follows from their product, as the danger threshold does.

A strategy that spends c ticks per attempt licks only while safe, so it
catches 1 - DANGER_FRACTION flies every c / q ticks, treating successive licks
at one fly as independent, and pays for a watch on almost every tick.
"""

import numpy as np
from scipy import special

from . import constants as C

POS_NOISE_VAR = 10.0 ** (-2 * C.POS_DECIMALS) / 12


def hit_probability(G, amps, noise_cov):
    """P(|G (a, b) + noise| <= R) over random phases and Gaussian noise."""
    n = len(amps)
    c, d = G[:, :n] * amps, G[:, n:] * amps
    R = C.CATCH_RADIUS
    reach = np.sum(np.hypot(*c) + np.hypot(*d))
    reach += 8 * np.sqrt(np.trace(noise_cov))
    # The density is summed over a periodic box wide enough that no copy
    # of it reaches the disk around the origin.
    L = 2 * (reach + 2 * R)
    k = 2 * np.pi * np.fft.fftfreq(C.CF_GRID, L / C.CF_GRID)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    phi = np.exp(-0.5 * (noise_cov[0, 0] * kx ** 2 + noise_cov[1, 1] * ky ** 2
                         + 2 * noise_cov[0, 1] * kx * ky))
    for (cx, cy), (dx, dy) in zip(c.T, d.T):
        phi *= special.j0(np.hypot(kx * cx + ky * cy, kx * dx + ky * dy))
    kr = np.hypot(kx, ky)
    kr[0, 0] = 1.0
    disk = 2 * np.pi * R * special.j1(R * kr) / kr
    disk[0, 0] = np.pi * R ** 2
    return np.sum(phi * disk) / L ** 2


def readout(world, t):
    """Position at time t as a linear function of (a, b)."""
    u = world.P @ world.fly.V[:world.N]
    wt = world.fly.w * t
    return np.c_[u * np.cos(wt), -u * np.sin(wt)]


def window_hit(world, k):
    """Best linear guess of y(k) from rounded y(0), ..., y(k - 1)."""
    var = np.tile(world.fly_amps ** 2 / 2, 2)
    O = np.vstack([readout(world, t) for t in range(k)])
    H = readout(world, k)
    S_yy = (O * var) @ O.T + POS_NOISE_VAR * np.eye(2 * k)
    gain = np.linalg.solve(S_yy, O @ (H * var).T).T
    return hit_probability(H - gain @ O, world.fly_amps,
                           POS_NOISE_VAR * gain @ gain.T)


def last_seen_hit(world):
    """Lick where the fly was one tick earlier."""
    return hit_probability(readout(world, 1) - readout(world, 0),
                           world.fly_amps, POS_NOISE_VAR * np.eye(2))


def cycle(k):
    return max(k + 1, C.LICK_INTERVAL)


def ticks_per_catch(world):
    """Safe ticks per catch for the easy, medium and hard strategies."""
    easy = C.LICK_INTERVAL / last_seen_hit(world)
    q = {k: window_hit(world, k)
         for k in range(1, max(C.MAX_WINDOW, C.MEDIUM_WINDOW) + 1)}
    medium = cycle(C.MEDIUM_WINDOW) / q[C.MEDIUM_WINDOW]
    # Hard tries its first window on a new fly and a sure one after a miss.
    sure = next((k for k in range(1, C.MAX_WINDOW + 1)
                 if q[k] >= C.SURE_HIT), C.MAX_WINDOW)
    retry = cycle(sure) / q[sure]
    hard = min(cycle(k) + (1 - q[k]) * retry
               for k in range(1, C.MAX_WINDOW + 1))
    return easy, medium, hard


def intake(ticks):
    return (1 - C.DANGER_FRACTION) * C.CATCH_ENERGY / ticks - C.WATCH_COST


def drains(world):
    easy, medium, hard = (intake(t) for t in ticks_per_catch(world))
    return {"easy": C.EASY_DRAIN_SHARE * easy,
            "medium": np.sqrt(easy * medium),
            "hard": np.sqrt(medium * hard)}
