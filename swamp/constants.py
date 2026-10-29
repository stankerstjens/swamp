"""Every numeric choice the specification leaves open."""

# Fly block. Masses, and the ranges the couplings are drawn from.
M_R = 1.0
M_G = 100.0
A_DIAG = (0.4, 1.6)
A_OFFDIAG_SD = 0.15
B_DIAG = (0.3, 1.0)
B_OFFDIAG_SD = 0.15
C_SD = 0.3

# Population block.
POP_DIAG = (1.0, 2.0)
POP_LINK = (0.3, 1.0)
POP_STIFFNESS = 0.05
EQUILIBRIUM = (1e2, 1e3)
POP_DEVIATION = 0.1
POP_VEL_SCALE = 0.1

MAX_REDRAWS = 10_000

# Decimals written to the data matrix.
STATE_DECIMALS = 3
