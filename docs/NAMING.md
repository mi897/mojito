# Naming and equation-commenting conventions

Read this before adding or renaming variables, functions or arguments. The
goal: someone who knows robotics but has not seen this code can read a line of
maths and know what each name is, in what units and frame, without opening
another file.

## Rules

1. **Spell it out.** A name says what the thing is: `upper_length`, not `lu`;
   `joint_velocities`, not `qd`; `learning_rate`, not `lr`.
2. **Public arguments follow the vocabulary below, exactly.** The same concept
   has the same argument name in every function, so keyword calls and
   signatures read alike across modules.
3. **Do not encode shape in the name; say it in the docstring.** Write
   `joint_angles` and document `(..., 3)`, not `q3` or `q_arr`.
4. **Units go in the name only when two units can appear in the same scope**
   (`median_mm`, `foot_load_n`). Otherwise the default applies: metres,
   radians, seconds, newtons, N·m (see `CLAUDE.md`).
5. **Booleans read as a claim:** `is_right`, `contact`, `hit`, not `flag`.
6. **Functions are verbs or "quantity of" nouns:** `solve`, `analytic_ik`,
   `foot_loads`, `stability_margin`. A function that returns a transformed
   copy says so (`mirror_target`); one that changes its argument says so in
   the docstring.
7. **Private helpers (`_name`) follow the same rules.** Private does not mean
   shorthand is fine.

## When a variable is part of an equation

Short symbols are allowed only for the plain coordinates and the
documented exceptions below, and only when the equation is written out next to
the code. Whenever a line implements a formula:

- Give every quantity in the formula a descriptive variable name
  (`sin_abduction`, `extension`, `stall_torque`), not a one-letter symbol.
- Put the formula in the function's docstring (the whole derivation if it is
  not textbook) or in a comment directly above the lines, written in the same
  names or in a stated notation, for example
  `torque = kp (target - angle) - kd velocity`.
- If the docstring uses math symbols (`l_u`, `e`, `a`, `b`, `k`), say once
  what each stands for, and keep that notation the same within the module.
- Name intermediate quantities that have a physical meaning
  (`extension`, `hip_plus_knee`, `speed_fraction`) instead of inlining them,
  so they can be referred to in the explanation.
- State the *why* of a non-obvious step (a clamp, a sign, a placeholder that
  avoids dividing by zero), not only the what.
- A new equation still needs a test against something independent
  (`CLAUDE.md`); mention in the docstring which test checks it, when one does.

## Vocabulary

| Concept | Name | Shape / meaning |
| --- | --- | --- |
| Joint values of a leg | `joint_angles` | `(..., 3)` rad: abduction, hip pitch, knee |
| Joint values of any limb | `joint_values` | `(..., dof)`, radians or metres (prismatic) |
| Joint velocities | `joint_velocities` | rad/s |
| Foot or tip target in a limb frame | `target_position` | `(..., 3)` m |
| Foot positions in the body frame | `feet_body` | `(..., 4, 3)` or dict by limb name |
| Link lengths | `lengths` (leg code), `link_lengths` (generic chain/limb code) | `(..., 3)` / `(..., n_params)` m |
| Abduction offset, upper, lower length | `abduction_offset`, `upper_length`, `lower_length` | m |
| Leg extension in the leg plane | `extension` | m, distance from the hip |
| Model configuration | `config` | a `LegConfig` or `LimbSpec` |
| Learning rate | `learning_rate` | CLI flag keeps `--lr` as an alias |
| Times | `times` | seconds, scalar or `(T,)` |
| Time step | `dt` | seconds (standard, kept) |
| Count of things | `n_<things>`: `n_legs`, `n_samples`, `n_joints` | int |
| Jacobian (as a local) | `jac` | `(..., 3, dof)` |
| Servo state | `params` is the dataclass; `.params` on the model | `ServoParams`, `JointSensorParams`, `ImuParams` |
| Random generator | `rng` | `np.random.Generator` |
| Extra keyword options passed on | `**solver_options`, `**sampler_options` | not `**kw` |

## Allowed short names

These stay, because they are standard and documented, not because shorthand
is generally fine:

- `x, y, z` for Cartesian coordinates of a point, and `roll, pitch, yaw`.
- `dt`, `rng`, `eps`, `i/j/k` as loop indices over a small, obvious range.
- `W[i]`, `b[i]` and the keys `W0, b0, ...` of the weight files: they are the
  file format, shaped `(inputs, outputs)` and `(outputs,)`. Locals that hold a
  single layer's weights use `weight`, `bias`.
- `tau` for joint torque, with the relation `tau = J^T F` stated where it is
  computed (as in `statics.joint_torques`).

## Names that must not change casually

Renaming these breaks saved files or the interface between modules:

- Fields written into weight files and the embedded limb spec: `nominal`,
  `limits`, `length_tolerance`, `reach`, `mount_R`, `mount_t`, `n_params`,
  and the `.npz` keys (`W0..`, `b0..`, `spec`). A rename needs a loader for
  the old name and regenerated weights.
- `LegConfig` and `LimbSpec` must keep the same public methods
  (`forward(joint_angles, link_lengths)`, `jacobian(...)`), since networks
  take either.
- The two backends keep the same public signatures (`CLAUDE.md`).
- Command-line flags, JSON keys in `results/`, and result file names.

## Where this has and has not been applied

Applied to the numerical core: `leg.py`, `kinematics.py`, `gaits.py`,
`statics.py`, `base.py`, `model.py`, `torch_model.py`, `actuators.py`, `sensors.py`,
`body.py`, and the public arguments of `spec.py`'s `LimbSpec` and `Mirror`.

Not yet swept, so shorthand remains there: the loaders and checks in
`spec.py` and `urdf.py`, `mjcf.py` and `stand.py` internals, and the plotting
and experiment code in `scripts/` and `tests/` (local variables such as `q`,
`p`, `L` and `J`). Clean these up as you touch them, following the rules
above, rather than renaming in bulk with search-and-replace: a blanket `p`
rename already collided with pitch, loop variables and the Adam optimiser.
