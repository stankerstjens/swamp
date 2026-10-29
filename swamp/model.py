"""The hidden world: two independent blocks of M x'' = -K x."""

import numpy as np
from scipy import integrate, linalg, special

from . import constants as C

# Species in the order they join the web as S grows, and what each eats.
SPECIES = ["flies", "frogs", "snakes", "herons", "alligators", "fish",
           "algae", "dragonflies", "mosquitoes", "turtles"]
DIET = {
    "flies": ["algae"],
    "frogs": ["flies", "dragonflies", "mosquitoes"],
    "snakes": ["frogs", "fish"],
    "herons": ["frogs", "fish", "snakes"],
    "alligators": ["frogs", "fish", "herons", "snakes", "turtles"],
    "fish": ["flies", "mosquitoes", "dragonflies", "algae"],
    "algae": [],
    "dragonflies": ["flies", "mosquitoes"],
    "mosquitoes": ["algae"],
    "turtles": ["algae", "fish", "frogs"],
}
FLY, FROG = 0, 1


class Oscillator:
    """Exact solution of M x'' = -K x through its normal modes."""

    def __init__(self, masses, K):
        self.masses = masses
        w2, self.V = linalg.eigh(K, np.diag(masses))
        self.w = np.sqrt(w2)

    def modal(self, x0, v0):
        return self.V.T @ (self.masses * x0), self.V.T @ (self.masses * v0)

    def position(self, q0, qd0, t):
        wt = self.w * t
        return self.V @ (q0 * np.cos(wt) + qd0 / self.w * np.sin(wt))

    def sinusoids(self, q0, qd0, readout):
        """Frequencies and amplitudes of readout @ x(t), equal ones merged."""
        gain = readout @ self.V
        a, b = gain * q0, gain * qd0 / self.w
        order = np.argsort(self.w)
        w, a, b = self.w[order], a[order], b[order]
        new = np.r_[True, np.diff(w) > C.FREQ_MERGE_TOL * w[1:]]
        group = np.cumsum(new) - 1
        amps = np.hypot(np.bincount(group, a), np.bincount(group, b))
        return w[new], amps


def time_above(theta, amps):
    """Gil-Pelaez: fraction of time sum_k amps_k cos(.) spends above theta."""
    return _TimeAbove(amps)(theta)


class _TimeAbove:
    def __init__(self, amps):
        self.scale = np.sqrt(np.sum(amps ** 2) / 2)
        self.u = np.arange(0, C.GP_U_MAX, C.GP_STEP)
        self.prod = np.ones_like(self.u)
        for a in amps / self.scale:
            self.prod *= special.j0(a * self.u)

    def __call__(self, theta):
        th = theta / self.scale
        f = th * np.sinc(self.u * th / np.pi) * self.prod
        return 0.5 - integrate.simpson(f, x=self.u) / np.pi


def danger_threshold(amps, target=C.DANGER_FRACTION):
    F = _TimeAbove(amps)
    lo, hi = -np.sum(amps), np.sum(amps)
    for _ in range(60):
        mid = (lo + hi) / 2
        if F(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def draw_symmetric(rng, n, diag, offdiag_sd, mask=None):
    X = rng.normal(0, offdiag_sd, (n, n))
    if mask is not None:
        X = X * mask
    X = np.triu(X, 1)
    return X + X.T + np.diag(rng.uniform(*diag, n))


def draw_web(rng, S):
    names = SPECIES[:S]
    links = np.zeros((S, S), bool)
    for i, pred in enumerate(names):
        prey = [names.index(p) for p in DIET[pred] if p in names]
        if not prey:
            continue
        keep = rng.random(len(prey)) < C.POP_LINK_PROB
        keep[rng.integers(len(prey))] = True
        # Frogs always eat flies and frog predators always eat frogs.
        keep |= np.isin(prey, [FLY if pred == "frogs" else FROG])
        for j, k in zip(prey, keep):
            links[i, j] = links[j, i] = k
    eats_frogs = np.array(["frogs" in DIET[n] for n in names])
    return names, links, eats_frogs


class World:
    """All parameters and initial states, drawn from one seed."""

    def __init__(self, n_neurons, n_genes, n_species, seed):
        N, G, S = n_neurons, n_genes, n_species
        self.N, self.G, self.S = N, G, S
        rng = np.random.default_rng(seed)
        for _ in range(C.MAX_REDRAWS):
            if self._draw(rng):
                break
        else:
            raise RuntimeError("no acceptable draw")
        _, self.pressure_amps = self.pop.sinusoids(
            self.pop_q0, self.pop_qd0, self.h)
        self.theta = danger_threshold(self.pressure_amps)

    def _draw(self, rng):
        N, G, S = self.N, self.G, self.S
        A = draw_symmetric(rng, N, C.A_DIAG, C.A_OFFDIAG_SD)
        B = draw_symmetric(rng, G, C.B_DIAG, C.B_OFFDIAG_SD)
        c = rng.normal(0, C.C_SD, G)
        P = rng.normal(0, 1, (2, N))
        coupling = np.kron(np.eye(N), c[None, :])
        K_f = np.block([[A, coupling],
                        [coupling.T, np.kron(np.eye(N), B)]])
        masses_f = np.r_[np.full(N, C.M_R), np.full(N * G, C.M_G)]

        names, links, eats_frogs = draw_web(rng, S)
        masses_p = rng.uniform(*C.POP_MASS, S)
        signs = rng.choice([-1.0, 1.0], (S, S))
        offdiag = np.triu(signs * rng.uniform(*C.POP_LINK, (S, S)) * links, 1)
        K_p = C.POP_STIFFNESS * (offdiag + offdiag.T
                                 + np.diag(rng.uniform(*C.POP_DIAG, S)))
        h = eats_frogs * rng.uniform(*C.FROG_PREDATOR_WEIGHT, S)
        eq = np.exp(rng.uniform(*np.log(C.EQUILIBRIUM), S))

        fly_phase = rng.uniform(0, 2 * np.pi, N + N * G)
        dev = C.POP_DEVIATION * eq
        p0 = rng.normal(0, dev)
        v0 = rng.normal(0, dev * C.POP_VEL_SCALE)

        if np.linalg.eigvalsh(K_f).min() <= 0:
            return False
        if np.linalg.eigvalsh(K_p).min() <= 0:
            return False
        pop = Oscillator(masses_p, K_p)
        q0, qd0 = pop.modal(p0, v0)
        for i in range(S):
            if pop.sinusoids(q0, qd0, np.eye(S)[i])[1].sum() >= eq[i]:
                return False

        # Scale and shear P so that the fly's steps per tick have the same
        # size and no preferred direction in every world.
        self.fly = Oscillator(masses_f, K_f)
        self.fly_amps = np.sqrt(2 * C.FLY_MODE_ENERGY) / self.fly.w
        w2, U = np.linalg.eigh(P @ self._step_covariance() @ P.T)
        P = C.STEP_SD * U @ np.diag(w2 ** -0.5) @ U.T @ P

        self.A, self.B, self.c, self.P = A, B, c, P
        self.K_f, self.masses_f = K_f, masses_f
        self.fly_state0 = self._fly_state(fly_phase)
        self.names, self.links, self.h, self.eq = names, links, h, eq
        self.K_p, self.masses_p = K_p, masses_p
        self.pop, self.pop_q0, self.pop_qd0 = pop, q0, qd0
        self.p0, self.v0 = p0, v0
        return True

    def _step_covariance(self):
        """Time-averaged covariance of r(t + 1) - r(t)."""
        Vr, w = self.fly.V[:self.N], self.fly.w
        return (Vr * self.fly_amps ** 2 * (1 - np.cos(w))) @ Vr.T

    def _fly_state(self, phase):
        q = self.fly_amps * np.cos(phase)
        qd = -self.fly_amps * self.fly.w * np.sin(phase)
        return self.fly.V @ q, self.fly.V @ qd

    def random_fly_state(self, rng):
        """Every fly mode gets the same energy and a random phase."""
        return self._fly_state(rng.uniform(0, 2 * np.pi, len(self.fly.w)))

    def populations(self, t):
        return self.eq + self.pop.position(self.pop_q0, self.pop_qd0, t)

    def pressure(self, t):
        return self.h @ self.pop.position(self.pop_q0, self.pop_qd0, t)
