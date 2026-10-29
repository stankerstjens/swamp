"""Each difficulty against the three reference players.

The drains come from the theory of these strategies (swamp/theory.py); here
the players learn the world in the sandbox and play it.
"""

import itertools
import unittest

from swamp import constants as C
from swamp.game import Game
from swamp.players import PLAYERS, explore

SURVIVES = {
    "easy": ["easy", "medium", "hard"],
    "medium": ["medium", "hard"],
    "hard": ["hard"],
}
SETTINGS = [(N, G, S, 1000 + i) for i, (N, G, S) in enumerate(
    itertools.product(C.NEURONS, C.GENES, C.SPECIES))]


class TestDifficulty(unittest.TestCase):
    def test_reference_players(self):
        for N, G, S, seed in SETTINGS:
            book = explore(N, G, S, seed)
            for player, difficulty in itertools.product(PLAYERS, SURVIVES):
                with self.subTest(N=N, G=G, S=S, seed=seed, player=player,
                                  difficulty=difficulty):
                    game = Game(difficulty, N, G, S, seed)
                    PLAYERS[player](book).play(game)
                    self.assertNotEqual(game.over, "eaten")
                    alive = game.over is None
                    self.assertEqual(alive, player in SURVIVES[difficulty])


if __name__ == "__main__":
    unittest.main()
