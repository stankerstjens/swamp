"""Turn-based game: every command is one tick."""

import numpy as np

from . import constants as C
from .model import FLY, World
from .theory import drains

DIFFICULTIES = ["sandbox", "easy", "medium", "hard"]
COSTS = {"watch": C.WATCH_COST, "neural": C.NEURAL_COST,
         "genes": C.GENES_COST, "lick": 0.0}
HELP = "commands: watch | lick x y | neural | genes"


class Game:
    def __init__(self, difficulty, n_neurons, n_genes, n_species, seed):
        self.difficulty = difficulty
        self.world = World(n_neurons, n_genes, n_species, seed)
        self.rng = np.random.default_rng(
            [seed, DIFFICULTIES.index(difficulty)])
        self.sandbox = difficulty == "sandbox"
        self.energy = None if self.sandbox else C.START_ENERGY
        self.drain = 0.0 if self.sandbox else drains(self.world)[difficulty]
        self.t = 0
        self.last_lick = -C.LICK_INTERVAL
        self.over = None
        self._new_fly(*self.world.fly_state0, 0)

    def _new_fly(self, x, v, t):
        self.fly_q0, self.fly_qd0 = self.world.fly.modal(x, v)
        self.fly_born = t

    def fly_state(self, t):
        return self.world.fly.position(
            self.fly_q0, self.fly_qd0, t - self.fly_born)

    def fly_position(self, t):
        return self.world.P @ self.fly_state(t)[:self.world.N]

    def in_danger(self, t):
        return self.world.pressure(t) > self.world.theta

    def step(self, command):
        """Run one command. Returns None if it is not a command."""
        if self.over:
            return None
        words = command.split()
        if not words or words[0] not in COSTS:
            return None
        if words[0] == "lick":
            try:
                target = np.array([float(words[1]), float(words[2])])
            except (IndexError, ValueError):
                return None
        elif len(words) != 1:
            return None

        t, W = self.t, self.world
        out = {"time": t, "command": words[0]}
        x = self.fly_state(t)
        if words[0] == "watch":
            out["position"] = tuple(np.round(W.P @ x[:W.N], C.POS_DECIMALS))
            out["fly_count"] = int(round(W.populations(t)[FLY]))
        elif words[0] == "neural":
            out["rates"] = np.round(x[:W.N], C.STATE_DECIMALS)
        elif words[0] == "genes":
            out["genes"] = np.round(x[W.N:], C.STATE_DECIMALS).reshape(
                W.N, W.G)
        elif t - self.last_lick < C.LICK_INTERVAL:
            out["ready"] = False
        else:
            self.last_lick = t
            out["ready"] = True
            out["caught"] = bool(np.linalg.norm(target - W.P @ x[:W.N])
                                 <= C.CATCH_RADIUS)
            if out["caught"]:
                self._new_fly(*W.random_fly_state(self.rng), t + 1)
            out["danger"] = bool(self.in_danger(t))
            if out["danger"] and not self.sandbox:
                self.over = "eaten"

        if not self.sandbox:
            self.energy -= COSTS[words[0]] + self.drain
            if out.get("caught"):
                self.energy += C.CATCH_ENERGY
            if self.energy <= 0 and not self.over:
                self.over = "hunger"
        out["energy"] = self.energy
        out["over"] = self.over
        self.t += 1
        return out


def describe(out):
    head = f"t={out['time']}  energy="
    head += "--" if out["energy"] is None else f"{max(out['energy'], 0):.2f}"
    lines = [head]
    if "position" in out:
        x, y = out["position"]
        lines.append(f"fly at ({x:.{C.POS_DECIMALS}f}, "
                     f"{y:.{C.POS_DECIMALS}f}), {out['fly_count']} flies")
    if "rates" in out:
        lines.append("firing rates: " + " ".join(
            f"{r:.{C.STATE_DECIMALS}f}" for r in out["rates"]))
    if "genes" in out:
        for i, row in enumerate(out["genes"]):
            lines.append(f"neuron {i + 1} genes: " + " ".join(
                f"{g:.{C.STATE_DECIMALS}f}" for g in row))
    if out["command"] == "lick":
        if not out["ready"]:
            lines.append("Your tongue is not ready.")
        else:
            lines.append("Caught it!" if out["caught"] else "Missed.")
            if out["danger"] and out["over"] != "eaten":
                lines.append("You barely escaped!")
    if out["over"] == "eaten":
        lines.append("A predator ate you. Game over.")
    if out["over"] == "hunger":
        lines.append("You died of hunger. Game over.")
    return "\n".join(lines)


def ask(prompt, choices):
    while True:
        answer = input(prompt).strip()
        if answer in choices:
            return answer


def ask_int(prompt, lo, hi):
    choices = [str(i) for i in range(lo, hi + 1)]
    return int(ask(f"{prompt} ({lo}-{hi}): ", choices))


def main():
    difficulty = ask("difficulty (sandbox, easy, medium, hard): ",
                     DIFFICULTIES)
    N = ask_int("neurons", *C.NEURONS)
    G = ask_int("genes per neuron", *C.GENES)
    S = ask_int("species", *C.SPECIES)
    while True:
        try:
            seed = int(input("seed (0 or more): "))
        except ValueError:
            continue
        if seed >= 0:
            break
    game = Game(difficulty, N, G, S, seed)
    print(HELP)
    while not game.over:
        try:
            command = input("> ")
        except EOFError:
            break
        out = game.step(command)
        print(HELP if out is None else describe(out))
