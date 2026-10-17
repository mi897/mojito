[Back to the README](../README.md)

# Physics results: mass, standing and gaits

`python scripts/body/torque_envelope.py` writes `results/body/torque_envelope.json`: the
static joint torque needed to hold the body in a stance, and the largest total
mass a servo can hold. For the Stage 2 design placeholders (50, 150, 150 mm
links, 212.5 mm stand height, 300 × 150 mm shoulder spacing; larger than the
Stage 0–1 ones above) and the unverified listed stall torque of 1.86 N·m at 5 V, at
half of stall: 3.59 kg standing on four feet, 2.15 kg in the crawl, 1.79 kg in the trot.
This counts weight on vertical foot loads only. It leaves out leg and servo mass
(twelve listed servos are 0.72 kg alone), gait dynamics, friction and heat, so it
is an upper bound, not a mass the robot can walk with. Details and the
length-to-width result are in `docs/STAGE2.md`.

`python scripts/stand/stand_test.py` (needs MuJoCo) then holds the standing posture in
physics through the servo model and writes `results/stand/stand_test.json`. With the
same placeholders, 5 V and servo gains that are also placeholders, a 1.5 kg robot
with near-weightless legs stands with the servo torques within 0.5% of the static
calculation when that is given the measured foot loads. The feet carry the weight
unequally (3.55 N front, 3.81 N rear), so the equal-quarter figure above is up
to 8% off. The sweep stood up to 4 kg and fell at 4.5 kg with the default gain,
and stood up to 5.5 kg and fell at 6 kg with a stiffer one: below the 7.18 kg
static bound either way, and the answer moves with gains that are placeholders.

`python scripts/stand/sit_to_stand.py` rises from a 140 mm crouch to the 212.5 mm stand over 0.5 to 4 s and
writes `results/stand/sit_to_stand.json` and `.png`; `python scripts/stand/make_stand_demo.py` writes the animations
`results/stand/stand_demo.gif` and `results/stand/sit_to_stand_demo.gif`. The crouch is the hardest pose (25% more knee
torque per newton than standing). With fine command resolution a 2 s rise peaks where the crouch hold does, 0.54 of
the stall torque for 3 kg and 0.72 for 4 kg; 5 kg does not rise. The weight is not shared equally: 24% on each front foot and 26% on each rear foot
for near-weightless legs, 20% and 30% with the default placeholder masses.
The masses, gains and servo figures are placeholders, so these show how the model behaves, not what a built robot can do.

`python scripts/gait/gait_replay.py` (needs MuJoCo) replays the trot (1.7 s period, `gaits.TROT`) and crawl through the servo model with a free body and
writes `results/gait/gait_replay.json`. With the default placeholder masses (1.72 kg) all nine runs stay upright (tilt at most 6.6°). The trot uses the tuned settings (22 mm swing, 0.16 m stance, kp 50, kd 0.9; below), the crawl the original ones (kp 10, kd 0.2, 0.2125 m). The body does not reach the commanded speed: trot at 0.15 m/s
measured 0.132 m/s (it measured 0.100 m/s with the original 30 mm swing, 0.2125 m stance and kp 10), at 0.05 m/s 0.038 m/s (0.011), and at 0.30 m/s 0.249 m/s (0.248); marching in place no longer drifts (0.000 m/s, against -0.008) and the turn at 0.6 rad/s makes 0.535 rad/s (0.38); crawl at 0.04 m/s measured
0.028 m/s. Eight of the nine runs count as walked; the trot at 0.30 m/s does not, because the servos are saturated for 41% of the walk (more than the 20% limit; it did count as walked with the original settings, at 21%). Tilt at 0.15 m/s is 5.9° with the tuned trot against 5.2° with the original, and its peak torque is 0.89 of stall against 0.76; peak torque is 0.65 to 1.00 of the 5 V stall limit for the trot and 0.94 to 0.99 for the crawl. One run per case, placeholder everything. The trot period matters, and the original Stage 1 period of 0.5 s was not optimal in physics: the same 0.15 m/s command made 0.041 m/s, and a sweep from 0.5 to 2 s found 1.5 to 2 s did best, so 1.7 s is used (`scripts/gait/sweeps/trot_period_sweep.py`, `results/gait/sweeps/period/trot_period_sweep.json`). That holds for the 30 mm swing only: repeated over 3 gains, 3 masses and 2 swing heights (`scripts/gait/sweeps/trot_period_robustness.py`, `results/gait/sweeps/period/trot_period_robustness.json`), 1.7 s beat 0.5 s in 8 of 9 cells at 30 mm and in 1 of 9 at 60 mm, and 1.5 to 3 s is a plateau rather than a peak (`docs/STAGE2.md`, "Longer trot periods"). At the 1.7 s period, a 192-run sweep of duty, commanded speed, stance height, servo gain and swing height (`scripts/gait/sweeps/trot_1p7_parameter_sweep.py`, `results/gait/sweeps/parameters/trot_1p7_parameter_sweep.json`, plot `results/gait/sweeps/parameters/trot_1p7_parameter_effects.png`) found the swing height mattered most: the 30 mm swing made a median 0.64 of the commanded speed and the 60 mm swing 0.42, with all 9 falls at 60 mm (the best cell, 0.19 m stance, kp 30, 30 mm swing, made 0.89 of 0.2 m/s); details and limits in `docs/STAGE2.md`. Repeating each value 10 times with random mass (+-5%) and latency draws and changing one parameter at a time from that best cell (`scripts/gait/sweeps/trot_1p7_one_at_a_time.py`, `results/gait/sweeps/parameters/trot_1p7_one_at_a_time.json`): the speed ratio peaked at kp 45 (0.860, sd 0.003) and fell to 0.768 at kp 150; was flat in kd from 0.1 to 1.25 and went backward at kd 2.0; peaked at 20 to 26 mm swing (0.858 to 0.859) with 4.6° tilt at 20 mm against 7.9° at 30 mm; and rose to 0.870 at a 0.16 m stance, below which (0.15 m and lower) the commanded feet leave the joint limits. The spread is from those perturbations only and is small (sd 0.001 to 0.004, up to 0.008 at the lowest stances and 0.027 to 0.041 at kd 1.5 to 2.0, where the servos saturate); the combination of the best values was not run. Crossing kp (20 to 150, steps of 10) with kd (0 to 3) at 10 runs per cell (`scripts/gait/sweeps/trot_1p7_kp_kd_grid.py`, `results/gait/sweeps/servo_gains/trot_1p7_kp_kd_grid.json`) showed that the speed does not follow a bell in kd: it rises to a plateau, then drops sharply between kd 1.4 and 1.5 at every kp, and goes backward from kd 2. The best mean speed ratio was 0.865 at kp 50 to 60 (kd 0.7 to 1.4); at kp 150 the best kd (1.3) gave 0.778, only 0.01 above the tied rule's 0.768, and tilt at high kp (9.0 to 9.3 degrees) barely changed with kd, so the high-kp slowdown is not a damping choice in this model (`docs/STAGE2.md`). Running the best of each together (kp 50, kd 0.9, 22 mm swing, 0.16 m stance) against the baseline (kp 30, kd 0.5, 30 mm, 0.19 m) at four speeds and three masses, 10 runs each (`scripts/gait/sweeps/trot_1p7_combined.py`, `results/gait/sweeps/combined/trot_1p7_combined.json`): at 1.73 kg the speed ratio rose by 0.079, 0.063, 0.031 and 0.015 at 0.05, 0.1, 0.15 and 0.2 m/s and the tilt fell from 7.9 to 8.8 degrees to 5.8 to 6.0; at 1.04 kg by 0.037, 0.026, 0.012 and 0.000; the combined gain was smaller than the sum of the single-group gains in all 8 of those cells; at 3.02 kg the servos were at their stall limit in every run and there was no clear gain, and the 22 mm swing was worse than 30 mm (`docs/STAGE2.md`).`scripts/gait/gait_speed_diagnosis.py` splits the shortfall: the crawl is the soft servos (kp 10; it reaches its 0.04 m/s at kp 100); the trot is partly that and partly feet that still carry weight when commanded to swing, but at 1.7 s the body makes more than that second effect predicts and the reason was not traced (`docs/STAGE2.md`). Animations: `results/gait/trot_replay.gif`, `results/gait/crawl_replay.gif` (side view) and `results/gait/trot_iso.gif`, `results/gait/trot_turn_iso.gif`, `results/gait/crawl_iso.gif` (isometric, with a gait timing chart). The pace (left legs, then right) goes backward, slowly (-0.012 m/s at 0.1 m/s commanded at the 1.7 s period; -0.030 at the original 0.5 s), because the body rolls onto the feet it is lifting (`scripts/gait/pace_diagnosis.py`, `docs/STAGE2.md`). `python scripts/gait/try_gait.py --help` tries other timings (period, duty, offsets, lift order); how the timing is set is in `docs/STAGE2.md`.

## Running the live gait explorer on your machine

The physics runs in Python on your computer; the browser only draws and sends slider changes. Nothing leaves your machine (the
server listens on 127.0.0.1).

```bash
git clone https://github.com/mi897/mojito && cd mojito
./scripts/setup.sh
just live                                           # add --preset crawl, --walk, --port 9000 after it
```

Open http://127.0.0.1:8765. Press **Walk** to start and **Stand** to stop; **Reset** rebuilds the robot (needed after a fall); **Pause** freezes the physics.
Drag in the picture to rotate the view, use the wheel to zoom, double-click to reset the view; **Follow** keeps the body centred (the floor grid
stays fixed in the world, so you see the real speed). The strip under the picture shows, for the last 4 s, each leg's commanded stance (thin black bar)
against the weight it actually carries (coloured bar). The readouts show measured against commanded speed (in brackets), tilt, feet down and peak servo torque as a fraction of the stall torque.

Parameters (all apply while it runs; speed, yaw rate, lift and sway are eased in over about 0.3 s so the feet do not jump; the mass slider rebuilds the model and the robot stands again):

| Slider | What it sets |
|---|---|
| period | seconds per cycle |
| duty | fraction of the cycle a foot is on the ground |
| offset FL, FR, RL, RR | cycle phase at which each leg starts its stance; it lifts at offset + duty |
| lift, sway | swing height; crawl-style body lean (acts only when no two swings overlap, duty 0.75 or more) |
| forward, left, yaw | commanded body velocity |
| kp, kd | servo position-loop gains |
| mass | total mass; the placeholder masses scaled together |
| sim speed | physics speed relative to the wall clock |

Experiments that match results in `docs/STAGE2.md` (placeholder everything, one run each, so look for the same direction, not the same numbers):
trot at 0.15 m/s with kp 10, then kp 100 and kd 1 (the body should get much closer to the commanded speed); the **pace** preset (rolls onto the feet it lifts and goes backward);
lift 0.03 against 0.06 on the pace (roll about doubles); the **crawl** with sway 0 against 0.025; raising mass until the servos saturate (torque/stall reaches 1).
Playing with the sliders is exploration, not a measurement: for numbers use `scripts/gait/try_gait.py` and put them in `results/`.
