"""Three players that learn the swamp in the sandbox, with PCA at the core.

All three share one danger model. Delay-embed the fly count and take its
principal components: the number of components above the noise floor is
twice the number of population modes, and the shift between lagged axes gives
their frequencies. Predator pressure is a fixed linear readout of those modes,
learned from the "You barely escaped!" licks.

They differ in how they aim. Easy licks where it last saw the fly. Medium and
hard regress the next position on the principal components of a window of
recent positions, which span the fly's hidden state. Medium always watches
MEDIUM_WINDOW times, more than any fly needs. Hard picks the window that
catches the most flies per tick spent.
"""

import numpy as np
from scipy import optimize, sparse

from . import constants as C
from .game import Game

COUNT_NOISE = 1 / np.sqrt(12)
POS_NOISE = 10.0 ** -C.POS_DECIMALS / np.sqrt(12)


def delay_embed(series, k):
    """Rows are windows of k consecutive samples, flattened."""
    series = np.asarray(series, float).reshape(len(series), -1)
    return np.stack([series[i:i + k].ravel()
                     for i in range(len(series) - k + 1)])


def pca(X, noise_sd, center=True):
    """Mean and principal axes that stand above the rounding noise."""
    mu = X.mean(0) if center else np.zeros(X.shape[1])
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    n, d = X.shape
    floor = C.NOISE_FLOOR * noise_sd * (np.sqrt(n) + np.sqrt(d))
    return mu, Vt[s > floor], s


def sinusoid_basis(t, freqs):
    wt = np.outer(t, freqs)
    return np.c_[np.ones(len(t)), np.cos(wt), np.sin(wt)]


class Notebook:
    """Records what a sandbox session shows: counts, positions, escapes."""

    def __init__(self, game):
        self.game = game
        self.counts, self.labels, self.runs = [], [], [[]]

    def watch(self):
        out = self.game.step("watch")
        self.counts.append((out["time"], out["fly_count"]))
        self.runs[-1].append(out["position"])
        return out

    def lick(self, target):
        out = self.game.step(f"lick {target[0]} {target[1]}")
        if out["ready"]:
            self.labels.append((out["time"], out["danger"]))
        self.runs.append([])
        return out

    def segments(self, k):
        return [np.array(r) for r in self.runs if len(r) > k]


def explore(n_neurons, n_genes, n_species, seed):
    """A scripted sandbox session that all three players learn from."""
    book = Notebook(Game("sandbox", n_neurons, n_genes, n_species, seed))
    for _ in range(C.EXPLORE_CONTIGUOUS):
        book.watch()
    # Catch flies to see many of them, aiming with a long window.
    k = C.EXPLORE_WINDOW
    for _ in range(C.EXPLORE_FLIES):
        aim = Aim(book.segments(k), k)
        need = C.EXPLORE_FLY_TICKS
        for _ in range(C.EXPLORE_TRIES):
            while len(book.runs[-1]) < need:
                book.watch()
            if book.lick(aim(book.runs[-1][-k:]))["caught"]:
                break
            need = k
    # Lick where the fly was seen, as often as the tongue allows, to learn
    # when predators strike.
    while len(book.labels) < C.EXPLORE_LABELS:
        book.watch()
        out = book.watch()
        book.lick(out["position"])
    return book


class Danger:
    """Predator pressure as a linear readout of the fly count's modes."""

    def __init__(self, book):
        t, f = np.array(book.counts[:C.EXPLORE_CONTIGUOUS], float).T
        X = delay_embed(f - f.mean(), C.COUNT_EMBED)
        _, axes, self.spectrum = pca(X, COUNT_NOISE, center=False)
        # Shift invariance of the lag axes (ESPRIT): one lag step multiplies
        # each mode by exp(i w).
        shift = np.linalg.lstsq(axes[:, :-1].T, axes[:, 1:].T, rcond=None)[0]
        w = np.angle(np.linalg.eigvals(shift))
        w = np.sort(w[w > C.MIN_FREQ])

        # Polish the frequencies on every count seen, with the amplitudes
        # solved exactly for each trial set of frequencies.
        t, f = np.array(book.counts, float).T

        def residual(w):
            B = sinusoid_basis(t, w)
            return B @ np.linalg.lstsq(B, f, rcond=None)[0] - f

        self.freqs = optimize.least_squares(residual, w).x
        B = sinusoid_basis(t, self.freqs)
        coef = np.linalg.lstsq(B, f, rcond=None)[0]
        K = len(self.freqs)
        self.a, self.b = coef[1:K + 1], coef[K + 1:]
        self.amps = np.hypot(self.a, self.b)

        t_lab, y = np.array(book.labels, float).T
        x = self.features(t_lab)
        y = 2 * y - 1
        # Soft-margin linear classifier: hinge loss on y (rho . x - theta) >= 1
        # plus a small L1 penalty on rho = rho_plus - rho_minus.
        n = len(y)
        A = sparse.hstack([np.c_[-y[:, None] * x, y[:, None] * x, y],
                           -sparse.eye(n)])
        cost = np.r_[np.full(2 * K, C.L1_PENALTY), 0, np.full(n, 1 / n)]
        res = optimize.linprog(
            cost, A_ub=A, b_ub=-np.ones(n),
            bounds=[(0, None)] * 2 * K + [(None, None)] + [(0, None)] * n)
        self.rho = res.x[:K] - res.x[K:2 * K]
        self.theta = res.x[2 * K]
        score = self.score(np.arange(int(t_lab.max())))
        # Never lick where a score this high has ever meant danger.
        self.limit = (self.score(t_lab[y > 0]).min()
                      - C.DANGER_BUFFER * score.std())

    def features(self, t):
        wt = np.outer(np.atleast_1d(t), self.freqs)
        return (self.a * np.cos(wt) + self.b * np.sin(wt)) / self.amps

    def score(self, t):
        return self.features(t) @ self.rho - self.theta

    def safe(self, t):
        return self.score(t)[0] < self.limit


class Aim:
    """Principal component regression of the next position on the last k."""

    def __init__(self, segments, k):
        self.k = k
        W = np.concatenate([delay_embed(s, k + 1) for s in segments])
        X, y = W[:, :-2], W[:, -2:]
        self.mu, self.axes, self.spectrum = pca(X, POS_NOISE)
        Z = np.c_[(X - self.mu) @ self.axes.T, np.ones(len(X))]
        self.beta = np.linalg.lstsq(Z, y, rcond=None)[0]

    def __call__(self, window):
        return self.predict(np.ravel(window)[None])[0]

    def predict(self, X):
        Z = (X - self.mu) @ self.axes.T
        return np.c_[Z, np.ones(len(Z))] @ self.beta


def hit_rate(aim, segments):
    """How often the first k watches of a new fly aim within reach."""
    W = np.stack([s[:aim.k + 1].ravel() for s in segments])
    miss = np.linalg.norm(aim.predict(W[:, :-2]) - W[:, -2:], axis=1)
    return np.mean(miss <= C.CATCH_RADIUS)


class Player:
    """Watch the current fly, lick when the window is full and it is safe."""

    def __init__(self, book):
        self.danger = Danger(book)

    def window(self, misses):
        """Watches to take before the next lick at the current fly."""
        return 1

    def aim(self, window):
        return window[-1]

    def play(self, game, ticks=C.SURVIVAL_TICKS):
        seen, misses, last_lick = [], 0, -C.LICK_INTERVAL
        while not game.over and game.t < ticks:
            t, k = game.t, self.window(misses)
            if (len(seen) >= k and t - last_lick >= C.LICK_INTERVAL
                    and self.danger.safe(t)):
                x, y = self.aim(np.array(seen[-k:]))
                caught = game.step(f"lick {x} {y}")["caught"]
                seen, last_lick = [], t
                misses = 0 if caught else misses + 1
            else:
                seen.append(game.step("watch")["position"])
        return game


class EasyPlayer(Player):
    pass


class MediumPlayer(Player):
    def __init__(self, book):
        super().__init__(book)
        self.aim = Aim(book.segments(C.MEDIUM_WINDOW), C.MEDIUM_WINDOW)

    def window(self, misses):
        return C.MEDIUM_WINDOW


class HardPlayer(Player):
    """Shortest window worth trying on a new fly; a sure one after a miss."""

    def __init__(self, book):
        super().__init__(book)
        segs = book.segments(C.MAX_WINDOW)
        train, test = segs[::2], segs[1::2]
        ks = np.arange(1, C.MAX_WINDOW + 1)
        hits = np.array([hit_rate(Aim(train, k), test) for k in ks])
        sure = ks[np.argmax(hits >= hits.max())]
        ticks_per_fly = ks + 1 + (1 - hits) * (sure + 1)
        self.first = ks[np.argmin(ticks_per_fly)]
        self.sure = sure
        self.aims = {k: Aim(segs, k) for k in {self.first, self.sure}}

    def window(self, misses):
        return self.first if misses == 0 else self.sure

    def aim(self, window):
        return self.aims[len(window)](window)


PLAYERS = {"easy": EasyPlayer, "medium": MediumPlayer, "hard": HardPlayer}
