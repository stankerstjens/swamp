"""Every numeric choice the specification leaves open."""

# Settings the player may choose at startup (inclusive ranges).
NEURONS = (2, 6)
GENES = (1, 3)
SPECIES = (4, 10)

# Fly block. Masses, and the ranges the couplings are drawn from.
M_R = 1.0
M_G = 100.0
A_DIAG = (0.4, 1.6)
A_OFFDIAG_SD = 0.15
B_DIAG = (0.3, 1.0)
B_OFFDIAG_SD = 0.15
C_SD = 0.3
# Root mean square fly step per tick along each axis.
STEP_SD = 1.0

# Energy of every fly mode in a random fly state (t = 0 and after a catch).
FLY_MODE_ENERGY = 0.5

# Population block.
POP_MASS = (1.0, 3.0)
POP_DIAG = (1.0, 2.0)
POP_LINK = (0.3, 1.0)
POP_STIFFNESS = 0.05
POP_LINK_PROB = 0.5
EQUILIBRIUM = (1e4, 1e5)
POP_DEVIATION = 0.1
POP_VEL_SCALE = 0.1
FROG_PREDATOR_WEIGHT = (0.5, 1.5)

MAX_REDRAWS = 10_000

# Danger.
DANGER_FRACTION = 0.3
FREQ_MERGE_TOL = 1e-9
# Gil-Pelaez integration grid, in units of 1 / (sd of the pressure).
GP_U_MAX = 1e4
GP_STEP = 0.02

# Display precision (decimals).
POS_DECIMALS = 3
STATE_DECIMALS = 3

# Game rules.
CATCH_RADIUS = 0.4
LICK_INTERVAL = 3
START_ENERGY = 150.0
CATCH_ENERGY = 10.0
WATCH_COST = 0.005
NEURAL_COST = 2.0
GENES_COST = 2.0

# Drains (swamp/theory.py) sit between the strategies' theoretical intakes:
# medium and hard at the geometric mean of the intakes they separate, easy at
# this share of the easy intake. The theory treats licks at one fly as
# independent; with two neurons, licks every 3 ticks can resonate with the
# two fast fly modes and catch as little as 0.4 of what it predicts.
EASY_DRAIN_SHARE = 0.25
# A window that hits at least this often counts as sure.
SURE_HIT = 0.999
# Points per side of the grid for characteristic-function integrals.
CF_GRID = 256

# Players. Sandbox exploration, in ticks or counts.
EXPLORE_CONTIGUOUS = 3000
EXPLORE_WINDOW = 12
EXPLORE_FLIES = 100
EXPLORE_FLY_TICKS = 40
EXPLORE_TRIES = 5
EXPLORE_LABELS = 4000
# Fly-count delay embedding, and the lowest frequency kept as a mode.
COUNT_EMBED = 300
MIN_FREQ = 1e-3
# Singular values below NOISE_FLOOR times the rounding-noise edge are noise.
NOISE_FLOOR = 3.0
# Danger classifier: L1 penalty against the mean hinge loss, and the buffer
# below the lowest dangerous score, in sd of the score.
L1_PENALTY = 1e-6
DANGER_BUFFER = 0.05
# Watches before each lick: fixed for medium, at most MAX_WINDOW for hard.
MEDIUM_WINDOW = 14
MAX_WINDOW = 8

# Tests: a strategy survives if it is alive after this many ticks.
SURVIVAL_TICKS = 6000
