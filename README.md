# Swamp

A turn-based terminal game for teaching PCA. You are a frog. Flies buzz
around the swamp, and snakes, herons and alligators hunt you. Everything you
see comes from one hidden linear oscillator system, so the right moves hide
in a few principal components.

## Install

With [uv](https://docs.astral.sh/uv/), one command sets up a virtual
environment in `.venv`, installs numpy and scipy, and starts the game:

```bash
uv run swamp
```

Without uv, any Python 3.10 or newer will do:

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e .
```

```bash
.venv/bin/swamp
```

## Starting a game

At startup you choose a difficulty
(sandbox, easy, medium, hard), the number of neurons in the fly's brain (2 to
6), genes per neuron (1 to 3), species in the food web (4 to 10), and a seed
(0 or more). The seed and the three sizes fix the world, including where
every population starts, so a sandbox session teaches you the world you then
play. Only the flies that appear after each catch differ between
difficulties.

## Playing

Every command takes one tick.

- `watch` shows the fly's position and the fly count. It costs 0.005 energy.
- `lick x y` tries to catch the fly at (x, y). It works if the fly is within
  0.4 of that point. The tongue needs 3 ticks between licks. A lick sooner
  than that fails with "Your tongue is not ready." and still takes the tick.
- `neural` shows the fly's firing rates. It costs 2.
- `genes` shows the fly's gene levels, one row per neuron. It costs 2.

A catch is worth 10 energy and replaces the fly with a new one. You start
with 150 energy and lose a fixed drain every tick. At zero you starve.
Licking while predator pressure is above the danger threshold gets you eaten.
In the sandbox there is no energy, and a lick in danger prints "You barely
escaped!" instead. Each response starts with the tick at which the command
ran and your energy after it.

## The hidden world

The fly block is M_f x'' = -K_f x over firing rates r (N) and gene levels g
(N G), with K_f built from a neuron coupling A, a gene network B shared by
all neurons and a gene-to-rate coupling c, as in the specification. Gene
masses are 100 times the rate masses, so genetic modes are slow. A new fly,
at t = 0 or after a catch, gives every normal mode the same energy and a
random phase. The fly sits at P r. P starts as a random 2 x N matrix and is
then rescaled and sheared so that, averaged over time, the fly moves 1 unit
per tick along each axis in every direction. That keeps the catch radius
meaningful in every world.

The population block is M_p p'' = -K_p p over deviations from equilibrium,
with off-diagonal stiffness only on predator-prey links. The web grows in a
fixed order as you add species: flies, frogs, snakes, herons, alligators,
fish, algae, dragonflies, mosquitoes, turtles. Each species has a fixed list
of possible prey. Each possible link survives with probability 1/2, every
predator keeps at least one prey, frogs always eat flies and every frog
predator eats frogs. With two or more frog predators the web always branches.
Predator pressure is h^T p, with weights in [0.5, 1.5] on the frog predators.

Both blocks evolve exactly through their normal modes. Danger means pressure
above theta, where theta solves the Gil-Pelaez equation for a 30% time
fraction by bisection.

## Drains

The drains come from theory, computed for each world when the game starts
(`swamp/theory.py`). A fly is a sum of modes with known amplitudes and random
phases. Each reference strategy aims with a linear guess, so its miss is a
linear function of the phases plus rounding noise. Each mode adds a vector
that turns around an ellipse, whose characteristic function is a Bessel
function J0. Their product gives the chance q that a lick lands within the
catch radius, the same trick that gives the danger threshold. A strategy that
spends c ticks per attempt licks only during the 70% of time that is safe, so
it gains 0.7 x 10 q / c energy per tick, minus the cost of watching.

- Easy licks where it saw the fly one tick earlier, every 3 ticks.
- Medium predicts the fly from 14 watches with the best linear guess.
- Hard tries the window that costs the fewest expected ticks per catch,
  and the shortest window that hits 99.9% of the time after a miss.

The medium drain is the geometric mean of the easy and medium intakes, and
the hard drain the geometric mean of the medium and hard intakes. The easy
drain is a quarter of the easy intake. The theory treats successive licks at
one fly as independent. That holds on average, but with two neurons the
3-tick lick rhythm can resonate with the two fast fly modes, and the easy
frog then catches between 0.4 and 1.6 times what theory predicts. The medium
and hard players catch 0.9 to 1.3 times their theory, gaining because they
watch through danger and lick as soon as it ends. Typical drains:

| neurons | easy | medium | hard |
| --- | --- | --- | --- |
| 2 | 0.014 | 0.16 | 0.90-1.04 |
| 4 | 0.016-0.035 | 0.17-0.25 | 0.73-1.02 |
| 6 | 0.024-0.039 | 0.21-0.27 | 0.74-0.89 |

## The players

`swamp/players.py` has three players, one per difficulty. They are the
reference strategies above, but they know nothing of the model. Each learns
the world from a scripted sandbox session with the same seed and sizes, then
plays.

All three share the danger model. They delay-embed 3000 consecutive fly
counts in windows of 300 lags and take the principal components. The number
of components above the rounding-noise floor is twice the number of
population modes the flies feel, and the shift between lagged axes gives the
mode frequencies (ESPRIT). A least-squares fit to every count seen sharpens
them. Predator pressure is a fixed linear readout of the same modes, so the
players learn its direction from 4000 sandbox licks labelled by "You barely
escaped!", with a soft-margin linear program. They lick only below every
score at which they ever saw danger, minus a buffer. In 300 test worlds this
never let a frog lick in danger, and left it 65% to 70% of the time to lick.

They differ in how they aim. The easy player licks where it last saw the
fly. The medium and hard players regress the next position on the principal
components of a window of recent positions. Those components span the part
of the fly's hidden state the window can see, so the regression predicts the
fly once the window is long enough. The medium player always watches 14
times. The hard player measures how often each window length hits a newborn
fly, picks the one with the fewest ticks per catch, and switches to the
shortest sure window after a miss. In practice 2 to 4 watches pin down a fly
whose state has up to 48 dimensions, because the slow genetic modes barely
move between watches.

## Tests

```bash
uv run python -m unittest
```

or, without uv, `.venv/bin/python -m unittest`.

The tests check that accepted draws are stable, that populations stay
positive over 10^6 ticks, that the analytic danger fraction matches 2 x 10^6
simulated ticks, the core game rules, and that over 6000 ticks each
difficulty kills exactly the players it should, at the corners of the
settings.

## Choices

Every value lives in `swamp/constants.py`.

| constant | value | meaning |
| --- | --- | --- |
| NEURONS, GENES, SPECIES | 2-6, 1-3, 4-10 | settings the player may pick |
| M_R, M_G | 1, 100 | rate and gene masses |
| A_DIAG, A_OFFDIAG_SD | U(0.4, 1.6), 0.15 | neuron coupling A |
| B_DIAG, B_OFFDIAG_SD | U(0.3, 1.0), 0.15 | gene network B |
| C_SD | 0.3 | gene-to-rate coupling c |
| STEP_SD | 1 | rms fly step per tick per axis |
| FLY_MODE_ENERGY | 0.5 | energy per fly mode in a new fly |
| POP_MASS | U(1, 3) | population masses |
| POP_DIAG, POP_LINK | U(1, 2), +-U(0.3, 1) | K_p diagonal and links |
| POP_STIFFNESS | 0.05 | scale of K_p |
| POP_LINK_PROB | 0.5 | chance a possible link exists |
| EQUILIBRIUM | log-U(10^4, 10^5) | equilibrium population sizes |
| POP_DEVIATION, POP_VEL_SCALE | 0.1, 0.1 | initial deviations, rates |
| FROG_PREDATOR_WEIGHT | U(0.5, 1.5) | entries of h |
| MAX_REDRAWS | 10^4 | draws tried before giving up |
| DANGER_FRACTION | 0.3 | time in danger |
| FREQ_MERGE_TOL | 10^-9 | relative gap for equal frequencies |
| GP_U_MAX, GP_STEP | 10^4, 0.02 | Gil-Pelaez integration grid |
| POS_DECIMALS, STATE_DECIMALS | 3, 3 | displayed decimals |
| CATCH_RADIUS | 0.4 | catch radius |
| LICK_INTERVAL | 3 | ticks between licks |
| START_ENERGY, CATCH_ENERGY | 150, 10 | energy at start, per catch |
| WATCH_COST | 0.005 | energy per watch |
| NEURAL_COST, GENES_COST | 2, 2 | energy per neural or genes |
| EASY_DRAIN_SHARE | 0.25 | easy drain over easy intake |
| SURE_HIT | 0.999 | hit rate of a sure window |
| CF_GRID | 256 | grid for the hit integrals |
| EXPLORE_* | | length of the players' sandbox session |
| COUNT_EMBED, MIN_FREQ | 300, 10^-3 | players' fly-count embedding |
| NOISE_FLOOR | 3 | singular values kept above noise |
| L1_PENALTY, DANGER_BUFFER | 10^-6, 0.05 | players' danger classifier |
| MEDIUM_WINDOW, MAX_WINDOW | 14, 8 | watches per lick, medium and hard |
| SURVIVAL_TICKS | 6000 | test horizon |
