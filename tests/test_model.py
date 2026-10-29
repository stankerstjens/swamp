import itertools
import unittest

import numpy as np

from swamp import constants as C
from swamp.model import World

CORNERS = list(itertools.product(C.NEURONS, C.GENES, C.SPECIES))
WORLDS = []


def setUpModule():
    WORLDS.extend(World(N, G, S, 100 * i + seed)
                  for i, (N, G, S) in enumerate(CORNERS) for seed in range(3))


def sample(world, readout, ticks, chunk=100_000):
    """readout @ p(t) for t = 0, 1, ..., ticks - 1."""
    gain = readout @ world.pop.V
    for start in range(0, ticks, chunk):
        wt = np.outer(np.arange(start, start + chunk), world.pop.w)
        q = (world.pop_q0 * np.cos(wt)
             + world.pop_qd0 / world.pop.w * np.sin(wt))
        yield q @ gain.T


class TestDraws(unittest.TestCase):
    def test_accepted_draws_are_stable(self):
        for w in WORLDS:
            self.assertGreater(np.linalg.eigvalsh(w.K_f).min(), 0)
            self.assertGreater(np.linalg.eigvalsh(w.K_p).min(), 0)
            self.assertTrue(np.all(w.fly.w > 0) and np.all(w.pop.w > 0))

    def test_populations_stay_positive(self):
        for w in WORLDS:
            low = np.full(w.S, np.inf)
            for p in sample(w, np.eye(w.S), 1_000_000):
                low = np.minimum(low, p.min(0))
            self.assertTrue(np.all(w.eq + low > 0))

    def test_analytic_danger_fraction_matches_simulation(self):
        for w in WORLDS[::3]:
            above = sum(np.sum(x > w.theta)
                        for x in sample(w, w.h, 2_000_000))
            self.assertAlmostEqual(above / 2_000_000, C.DANGER_FRACTION,
                                   delta=0.005)


if __name__ == "__main__":
    unittest.main()
