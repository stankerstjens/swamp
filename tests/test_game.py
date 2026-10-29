import unittest

import numpy as np

from swamp import constants as C
from swamp.game import Game


def first_tick(game, danger):
    return next(t for t in range(10_000) if game.in_danger(t) == danger)


def wait_until(game, t):
    while game.t < t:
        game.step("watch")


class TestRules(unittest.TestCase):
    def test_catch_resets_fly_but_not_populations(self):
        game = Game("sandbox", 3, 2, 6, 1)
        before = game.fly_state(5)
        wait_until(game, 3)
        x, y = game.fly_position(3)
        out = game.step(f"lick {x} {y}")
        self.assertTrue(out["caught"])
        self.assertFalse(np.allclose(game.fly_state(5), before))
        self.assertTrue(np.allclose(game.world.populations(5),
                                    Game("sandbox", 3, 2, 6, 1)
                                    .world.populations(5)))

    def test_tongue_needs_rest(self):
        game = Game("sandbox", 3, 2, 6, 1)
        self.assertTrue(game.step("lick 0 0")["ready"])
        for _ in range(C.LICK_INTERVAL - 2):
            game.step("watch")
        self.assertFalse(game.step("lick 0 0")["ready"])
        self.assertTrue(game.step("lick 0 0")["ready"])

    def test_licking_in_danger(self):
        sandbox = Game("sandbox", 3, 2, 6, 1)
        wait_until(sandbox, first_tick(sandbox, True))
        self.assertTrue(sandbox.step("lick 0 0")["danger"])
        self.assertIsNone(sandbox.over)
        hard = Game("hard", 3, 2, 6, 1)
        wait_until(hard, first_tick(hard, True))
        self.assertEqual(hard.step("lick 0 0")["over"], "eaten")

    def test_hunger(self):
        game = Game("easy", 3, 2, 6, 1)
        while not game.over:
            game.step("genes")
        self.assertEqual(game.over, "hunger")
        self.assertIsNone(game.step("watch"))


if __name__ == "__main__":
    unittest.main()
