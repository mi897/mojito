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

`python scripts/sim/stand_test.py` (needs MuJoCo) then holds the standing posture in
physics through the servo model and writes `results/stand/stand_test.json`. With the
same placeholders, 5 V and servo gains that are also placeholders, a 1.5 kg robot
with near-weightless legs stands with the servo torques within 0.5% of the static
calculation when that is given the measured foot loads. The feet carry the weight
unequally (3.55 N front, 3.81 N rear), so the equal-quarter figure above is up
to 8% off. The sweep stood up to 4 kg and fell at 4.5 kg with the default gain,
and stood up to 5.5 kg and fell at 6 kg with a stiffer one: below the 7.18 kg
static bound either way, and the answer moves with gains that are placeholders.

`python scripts/sim/sit_to_stand.py` rises from a 140 mm crouch to the 212.5 mm stand over 0.5 to 4 s and
writes `results/stand/sit_to_stand.json` and `.png`; `python scripts/sim/make_stand_demo.py` writes the animations
`results/stand/stand_demo.gif` and `results/stand/sit_to_stand_demo.gif`. The crouch is the hardest pose (25% more knee
torque per newton than standing). With fine command resolution a 2 s rise peaks where the crouch hold does, 0.54 of
the stall torque for 3 kg and 0.72 for 4 kg; 5 kg does not rise. The weight is not shared equally: 24% on each front foot and 26% on each rear foot
for near-weightless legs, 20% and 30% with the default placeholder masses.
The masses, gains and servo figures are placeholders, so these show how the model behaves, not what a built robot can do.

`python scripts/sim/gait_replay.py` (needs MuJoCo) replays the trot (1.7 s period, `gaits.TROT_SLOW`) and crawl through the servo model with a free body and
writes `results/gait/gait_replay.json`. With the default placeholder masses (1.72 kg) all nine runs stay upright (tilt at most
12.1°, in the turning trot). The body does not reach the commanded speed: trot at 0.15 m/s
measured 0.100 m/s, at 0.30 m/s 0.248 m/s, and marching in place drifted back at 0.008 m/s; crawl at 0.04 m/s measured
0.028 m/s. Peak torque is 0.61 to 0.93 of the 5 V stall limit for the trot and 0.94 to 0.99 for the crawl. One run per case, placeholder everything. The trot period matters: at the Stage 1 period of 0.5 s the same 0.15 m/s command made 0.041 m/s, and 1.5 to 2 s did best (`scripts/sim/trot_period_sweep.py`, `results/gait/trot_period_sweep.json`, finding in `docs/STAGE2.md`, "Longer trot periods"). `scripts/sim/gait_speed_diagnosis.py` splits the shortfall: the crawl is the soft servos (kp 10; it reaches its 0.04 m/s at kp 100); the trot is partly that and partly feet that still carry weight when commanded to swing, but at 1.7 s the body makes more than that second effect predicts and the reason was not traced (`docs/STAGE2.md`). Animations: `results/gait/trot_replay.gif`, `results/gait/crawl_replay.gif` (side view) and `results/gait/trot_iso.gif`, `results/gait/trot_turn_iso.gif`, `results/gait/crawl_iso.gif` (isometric, with a gait timing chart). The pace (left legs, then right) goes backward because the body rolls onto the feet it is lifting (`scripts/sim/pace_diagnosis.py`, `docs/STAGE2.md`). `python scripts/sim/try_gait.py --help` tries other timings (period, duty, offsets, lift order); how the timing is set is in `docs/STAGE2.md`.

## Running the live gait explorer on your machine

The physics runs in Python on your computer; the browser only draws and sends slider changes. Nothing leaves your machine (the
server listens on 127.0.0.1).

```bash
git clone https://github.com/mi897/mojito && cd mojito
git checkout stage-2-trot-crawl-physics-replay      # until it is merged to main
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
Playing with the sliders is exploration, not a measurement: for numbers use `scripts/sim/try_gait.py` and put them in `results/`.
