"""What both network backends share: limb configuration, mirroring, the weight file format.

`IKNetBase` is subclassed by the NumPy backend (`model.IKNet`) and the PyTorch
backend (`torch_model.TorchIKNet`). `reconcile` checks that saved weights fit
a URDF-derived limb. Nothing here depends on PyTorch.
"""
from __future__ import annotations

import json

import numpy as np

from ..geometry.limb_model import LimbModel
from ..geometry.spec import LimbSpec, SpecMismatch
from ..robot import leg


def reconcile(file_cfg: LimbModel, spec: LimbSpec, where: str = "weights") -> LimbSpec:
    """Check that weights trained for `file_cfg` fit the URDF-derived `spec`, and return the config to use.

    The weights stay authoritative for input scaling (nominal lengths,
    tolerance, reach); the URDF has to agree on structure.

    For legacy LegConfig weights the URDF limb's forward kinematics is compared
    with the leg model through `leg.is_leg`: 16 poses drawn over the placeholder
    joint limits and lengths, not (as before) 32 poses over the weights' own
    limits. The weights' limits are still compared with the URDF's separately.
    """
    if isinstance(file_cfg, LimbSpec):
        diffs = file_cfg.describe_mismatch(spec)
        if diffs:
            raise SpecMismatch(f"{where} do not match the URDF limb {spec.name!r}:\n  " + "\n  ".join(diffs))
        return file_cfg
    # Legacy weights: a LegConfig. The URDF limb must be the same 3-joint leg.
    diffs = []
    if (spec.dof, spec.n_params) != (3, 3):
        diffs.append(f"weights are for 3 joints and 3 lengths, the URDF limb has {spec.dof} and {spec.n_params}")
    else:
        if not np.allclose(spec.limits, file_cfg.limits, atol=1e-9):
            diffs.append(f"joint limits: {file_cfg.limits.tolist()} in weights vs {spec.limits.tolist()} in URDF")
        if not leg.is_leg(spec):
            diffs.append("forward kinematics of the URDF limb differs from the leg model the weights were trained on")
    if diffs:
        raise SpecMismatch(f"{where} do not match the URDF limb {spec.name!r}:\n  " + "\n  ".join(diffs))
    return spec.with_normalisation(file_cfg.nominal, file_cfg.length_tolerance, file_cfg.reach)


class IKNetBase:
    """What every backend shares: configuration, mirroring, and the weight file format.

    A subclass provides predict(target_position, lengths), train_step(target_position, lengths, learning_rate),
    state() and _set_weights(W, b). Weights travel as NumPy arrays with W[i]
    shaped (inputs, outputs), whatever the backend stores internally.
    """

    backend = None

    def __init__(self, config: LimbModel | None = None, hidden=(128, 128, 128)):
        self.config: LimbModel = config or leg.LegConfig()
        self.hidden = tuple(hidden)
        self.in_dim = 3 + self.config.n_params
        self.out_dim = self.config.dof
        limits = self.config.limits
        # Output squashing: joint_angles = joint_mid + joint_half_range * tanh(z) always lies inside the limits.
        self.joint_mid = limits.mean(axis=1)  # centre of each joint's range
        self.joint_half_range = 0.5 * (limits[:, 1] - limits[:, 0])  # half-width of each joint's range

    def predict_leg(self, target_position, lengths, right: bool = False, mirror=None):
        """As `predict`, for either side of the robot.

        `mirror` is the `limb_model.Mirror` relating the limb to this network's canonical
        limb; by default the left/right leg mirror.
        """
        if not right:
            return self.predict(target_position, lengths)
        mirror = mirror or self.config.mirror
        return mirror.angles(self.predict(mirror.target(target_position), lengths))

    def weights(self):
        """(list of W, list of b) as NumPy arrays."""
        d = self.state()
        n = sum(1 for k in d if k.startswith("W"))
        return [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)]

    def _config_state(self) -> dict:
        d = dict(nominal=self.config.nominal, limits=self.config.limits,
                 length_tolerance=np.array(self.config.length_tolerance))
        if isinstance(self.config, LimbSpec):
            d["spec"] = np.array(json.dumps(self.config.to_dict()))
        return d

    def save(self, path):
        np.savez(path, **self.state())

    @classmethod
    def load(cls, path, spec: LimbSpec | None = None, **kwargs):
        """Load weights. Pass the URDF-derived `spec` to verify they belong to that limb.

        Raises spec.SpecMismatch, listing every difference, if they do not.
        """
        d = np.load(path)
        n_layers = sum(1 for k in d.files if k.startswith("W"))
        if "spec" in d.files:
            config = LimbSpec.from_dict(json.loads(str(d["spec"])))
        else:  # weights from before limbs were described by URDFs
            config = leg.LegConfig(d["nominal"], d["limits"], float(d["length_tolerance"]))
        if spec is not None:
            config = reconcile(config, spec, str(path))
        hidden = tuple(d[f"W{i}"].shape[1] for i in range(n_layers - 1))
        net = cls(config, hidden, **kwargs)
        net._set_weights([d[f"W{i}"] for i in range(n_layers)], [d[f"b{i}"] for i in range(n_layers)])
        return net

    def to_json(self, digits: int = 6) -> str:
        """Compact export for running the network outside Python."""
        r = lambda a: np.round(np.asarray(a, dtype=float), digits).tolist()
        W, b = self.weights()
        return json.dumps(
            {
                "W": [r(w) for w in W],
                "b": [r(x) for x in b],
                "nominal": r(self.config.nominal),
                "limits": r(self.config.limits),
                "tolerance": self.config.length_tolerance,
                "reach": self.config.reach,
                **({"spec": self.config.to_dict()} if isinstance(self.config, LimbSpec) else {}),
            },
            separators=(",", ":"),
        )
