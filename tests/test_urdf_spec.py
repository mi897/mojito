"""URDF parsing, the spec/alignment layer, and generic kinematics.

    python -m unittest discover tests
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from mojito import IKNet, body, kinematics, leg, robots, spec as specmod, urdf  # noqa: E402
from mojito.model import SpecMismatch  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
WEIGHTS = os.path.join(ROOT, "weights", "ik_leg.npz")


def random_chain_urdf(rng, dof, prismatic_at=(), fixed_rot=True):
    """A serial chain with random axes, offsets and fixed rotations: no structure to rely on."""
    joints, links = {}, ["base"]
    for i in range(dof):
        axis = rng.normal(size=3)
        kind = "prismatic" if i in prismatic_at else "revolute"
        j = urdf.Joint(f"j{i}", kind, links[-1], f"l{i}", xyz=tuple(rng.uniform(-0.2, 0.2, 3)) if i else (0.1, 0.0, 0.0),
                       rpy=tuple(rng.uniform(-1, 1, 3)) if fixed_rot else (0, 0, 0), axis=tuple(axis),
                       lower=-1.0, upper=1.0)
        joints[j.name] = j
        links.append(f"l{i}")
    joints["tip"] = urdf.Joint("tip", "fixed", links[-1], "tipl", xyz=tuple(rng.uniform(0.05, 0.2, 3)))
    links.append("tipl")
    return urdf.URDFModel("rand", links, joints)


def limb_of(model, base="base", tip=None):
    return specmod.build_limb(model, base, tip or model.leaves(base)[0])


class TestParser(unittest.TestCase):
    def test_round_trip(self):
        m = robots.quadruped_urdf()
        again = urdf.parse(urdf.write(m))
        self.assertEqual(again.links, m.links)
        self.assertEqual(again.joints, m.joints)
        self.assertEqual(urdf.write(again), urdf.write(m))

    def test_unknown_elements_survive_a_write(self):
        text = ('<robot name="r"><link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>'
                '<link name="b"/><joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
                '<origin xyz="0 0 1"/><axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint></robot>')
        m = urdf.parse(text)
        m.joints["j"].xyz = (0.0, 0.0, 2.0)
        out = urdf.write(m)
        self.assertIn("<visual>", out)
        self.assertEqual(urdf.parse(out).joints["j"].xyz, (0.0, 0.0, 2.0))

    def test_file_and_string_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.urdf")
            urdf.write(robots.planar_urdf(), path)
            self.assertEqual(urdf.parse(path).joints.keys(), robots.planar_urdf().joints.keys())

    def test_errors_name_the_problem(self):
        bad = {
            "not xml": ("<robot", "not valid XML"),
            "wrong root": ("<foo/>", "expected <robot>"),
            "missing link": ('<robot><link name="a"/><joint name="j" type="fixed"><parent link="a"/><child link="zzz"/></joint></robot>', "'zzz'"),
            "two parents": ('<robot><link name="a"/><link name="b"/><link name="c"/>'
                            '<joint name="j1" type="fixed"><parent link="a"/><child link="c"/></joint>'
                            '<joint name="j2" type="fixed"><parent link="b"/><child link="c"/></joint></robot>', "two parent joints"),
            "bad type": ('<robot><link name="a"/><link name="b"/><joint name="j" type="wobbly"><parent link="a"/><child link="b"/></joint></robot>', "wobbly"),
            "bad number": ('<robot><link name="a"/><link name="b"/><joint name="j" type="fixed"><parent link="a"/><child link="b"/><origin xyz="1 x 3"/></joint></robot>', "xyz"),
            "cycle": ('<robot><link name="r"/><link name="a"/><link name="b"/>'
                      '<joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>'
                      '<joint name="j2" type="fixed"><parent link="b"/><child link="a"/></joint></robot>', "cycle"),
        }
        for label, (text, needle) in bad.items():
            with self.subTest(label):
                with self.assertRaises(urdf.URDFError) as cm:
                    urdf.parse(text)
                self.assertIn(needle, str(cm.exception))


class TestKinematics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_generic_fk_matches_the_hand_written_leg(self):
        limb = specmod.build_limb(robots.leg_urdf(), "body", "L_foot")
        q = self.rng.uniform(leg.JOINT_LIMITS[:, 0], leg.JOINT_LIMITS[:, 1], size=(500, 3))
        L = leg.NOMINAL_LENGTHS * (1 + self.rng.uniform(-0.2, 0.2, size=(500, 3)))
        np.testing.assert_allclose(limb.forward(q, L), leg.forward(q, L), atol=1e-12)
        np.testing.assert_allclose(limb.jacobian(q, L), leg.jacobian(q, L), atol=1e-12)

    def test_parameters_follow_urdf_order(self):
        limb = specmod.build_limb(robots.leg_urdf(), "body", "L_foot")
        np.testing.assert_allclose(limb.nominal, leg.NOMINAL_LENGTHS)
        self.assertEqual(limb.param_joints, ("L_hip_pitch", "L_knee", "L_foot"))

    def test_jacobian_matches_finite_differences_on_random_chains(self):
        for dof, pris in ((2, ()), (4, ()), (6, ()), (5, (2,))):
            with self.subTest(dof=dof, prismatic=pris):
                limb = limb_of(random_chain_urdf(self.rng, dof, pris))
                self.assertEqual(limb.dof, dof)
                q = self.rng.uniform(-0.8, 0.8, size=(20, dof))
                p = limb.nominal * (1 + self.rng.uniform(-0.1, 0.1, size=(20, limb.n_params)))
                J = limb.jacobian(q, p)
                self.assertEqual(J.shape, (20, 3, dof))
                h = 1e-6
                for j in range(dof):
                    d = np.zeros(dof)
                    d[j] = h
                    fd = (limb.forward(q + d, p) - limb.forward(q - d, p)) / (2 * h)
                    np.testing.assert_allclose(J[..., j], fd, atol=1e-7)

    def test_fixed_joints_before_the_first_joint_become_the_mount(self):
        m = robots.leg_urdf()
        m.links.insert(1, "plate")
        m.joints = {"mount": urdf.Joint("mount", "fixed", "body", "plate", xyz=(1.0, 2.0, 3.0), rpy=(0, 0, 0.5)),
                    **{k: v for k, v in m.joints.items()}}
        m.joints["L_abduction"].parent = "plate"
        limb = specmod.build_limb(m, "body", "L_foot")
        np.testing.assert_allclose(limb.mount_t, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(limb.mount_R, specmod.rpy_matrix((0, 0, 0.5)))
        q = np.array([[0.1, 0.5, -1.0]])
        np.testing.assert_allclose(limb.forward(q, leg.NOMINAL_LENGTHS), leg.forward(q, leg.NOMINAL_LENGTHS), atol=1e-12)

    def test_numeric_ik_reaches_targets_on_a_chain_with_no_closed_form(self):
        limb = limb_of(random_chain_urdf(self.rng, 4, fixed_rot=False))
        q_true = self.rng.uniform(-0.5, 0.5, size=(200, 4))
        p = limb.forward(q_true, limb.nominal)
        q = limb.numeric_ik(p, limb.nominal, iters=200)
        err = np.linalg.norm(limb.forward(q, limb.nominal) - p, axis=1)
        self.assertGreater(np.mean(err < 1e-6), 0.9)

    def test_numeric_ik_matches_closed_form_on_the_leg(self):
        limb = specmod.build_limb(robots.leg_urdf(), "body", "L_foot")
        rng = np.random.default_rng(5)
        p, L, q_true = leg.sample_batch(rng, 300, leg.LegConfig())
        q = limb.numeric_ik(p, L, iters=100)
        self.assertGreater(np.mean(np.linalg.norm(limb.forward(q, L) - p, axis=1) < 1e-6), 0.95)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_numpy_and_torch_agree(self):
        import torch

        limb = limb_of(random_chain_urdf(self.rng, 5, (1,)))
        q = self.rng.uniform(-0.8, 0.8, size=(10, 5))
        p = limb.nominal * (1 + self.rng.uniform(-0.1, 0.1, size=(10, limb.n_params)))
        tq, tp = torch.tensor(q, dtype=torch.float64, requires_grad=True), torch.tensor(p, dtype=torch.float64)
        np.testing.assert_allclose(limb.forward(tq, tp).detach().numpy(), limb.forward(q, p), atol=1e-12)
        np.testing.assert_allclose(limb.jacobian(tq, tp).detach().numpy(), limb.jacobian(q, p), atol=1e-12)
        # autograd of fk agrees with the analytic Jacobian
        auto = torch.autograd.functional.jacobian(lambda x: limb.forward(x, tp[0]), tq[0])
        np.testing.assert_allclose(auto.numpy(), limb.jacobian(q[0], p[0]), atol=1e-9)


class TestSpecGrouping(unittest.TestCase):
    def test_quadruped_is_one_group_of_four(self):
        r = specmod.from_urdf(robots.quadruped_urdf(), robots.QUADRUPED_MANIFEST)
        self.assertEqual(list(r.groups), ["leg"])
        self.assertEqual([m.name for m in r.groups["leg"].members], ["FL", "FR", "RL", "RR"])
        self.assertEqual([m.mirrored for m in r.groups["leg"].members], [False, True, False, True])
        self.assertEqual(r.validate(), [])

    def test_hexapod_is_one_group_of_six(self):
        r = specmod.from_urdf(robots.hexapod_urdf(), robots.HEXAPOD_MANIFEST)
        self.assertEqual(len(r.groups), 1)
        self.assertEqual(len(r.groups["leg"].members), 6)

    def test_no_manifest_discovers_limbs_and_mirrors(self):
        r = specmod.from_urdf(robots.quadruped_urdf())
        self.assertEqual(len(r.limbs), 4)
        self.assertEqual(len(r.groups), 1)
        self.assertEqual(sum(m.mirrored for g in r.groups.values() for m in g.members), 2)

    def test_mixed_robot_has_one_group_per_limb_type(self):
        m = robots.quadruped_urdf()
        m.links += ["tail1", "tail2", "tail3"]
        for j in (urdf.Joint("tail_a", "revolute", "body", "tail1", xyz=(-0.2, 0, 0), axis=(0, 1, 0), lower=-1, upper=1),
                  urdf.Joint("tail_b", "revolute", "tail1", "tail2", xyz=(-0.05, 0, 0), axis=(0, 1, 0), lower=-1, upper=1),
                  urdf.Joint("tail_tip", "fixed", "tail2", "tail3", xyz=(-0.05, 0, 0))):
            m.joints[j.name] = j
        r = specmod.from_urdf(m)
        self.assertEqual(len(r.groups), 2)
        self.assertEqual(sorted(len(g.members) for g in r.groups.values()), [1, 4])

    def test_different_limits_do_not_share_a_network(self):
        m = robots.quadruped_urdf()
        m.joints["RL_knee"].lower = -2.0
        r = specmod.from_urdf(m)
        self.assertEqual(len(r.groups), 2)

    def test_declared_group_that_differs_explains_why(self):
        m = robots.quadruped_urdf()
        m.joints["RL_knee"].lower = -2.0
        manifest = {"limbs": [{"name": "FL", "tip": "FL_foot", "group": "leg"},
                              {"name": "RL", "tip": "RL_foot", "group": "leg"}]}
        with self.assertRaises(specmod.SpecError) as cm:
            specmod.from_urdf(m, manifest)
        self.assertIn("limits", str(cm.exception))

    def test_a_false_mirror_claim_is_rejected(self):
        m = robots.quadruped_urdf()
        m.joints["FR_knee"].xyz = (0.0, 0.0, -0.12)
        m.joints["FR_knee"].axis = (1.0, 0.0, 0.0)  # knee turns about the wrong axis
        manifest = {"limbs": [{"name": "FL", "tip": "FL_foot"}, {"name": "FR", "tip": "FR_foot", "mirror_of": "FL"}]}
        with self.assertRaisesRegex(specmod.SpecError, "not the y-mirror"):
            specmod.from_urdf(m, manifest)

    def test_mirror_relation_holds_numerically(self):
        r = specmod.from_urdf(robots.quadruped_urdf(), robots.QUADRUPED_MANIFEST)
        left, right = r.limbs["FL"], r.limbs["FR"]
        mirror = r.member("FR").mirror
        rng = np.random.default_rng(1)
        q = rng.uniform(left.limits[:, 0], left.limits[:, 1], size=(50, 3))
        np.testing.assert_allclose(right.forward(mirror.angles(q), left.nominal), mirror.target(left.forward(q, left.nominal)), atol=1e-12)
        self.assertEqual(mirror.sign_q, (-1.0, 1.0, 1.0))  # abduction flips, as in leg.mirror_angles

    def test_unsupported_structure_is_refused(self):
        m = robots.planar_urdf()
        m.joints["knee"].mimic = "hip"
        with self.assertRaisesRegex(specmod.SpecError, "mimic"):
            specmod.from_urdf(m)
        m = robots.quadruped_urdf()
        m.links.append("antenna")
        m.joints["ant"] = urdf.Joint("ant", "fixed", "FL_thigh", "antenna", xyz=(0, 0, 0.1))
        with self.assertRaisesRegex(specmod.SpecError, "branches"):
            specmod.from_urdf(m, robots.QUADRUPED_MANIFEST)

    def test_signature_ignores_lengths_but_not_structure(self):
        a = specmod.build_limb(robots.leg_urdf(), "body", "L_foot")
        b = specmod.build_limb(robots.leg_urdf(lengths=(0.05, 0.14, 0.11)), "body", "L_foot")
        self.assertEqual(a.signature, b.signature)
        c = robots.leg_urdf()
        c.joints["L_knee"].upper = -0.5
        self.assertNotEqual(a.signature, specmod.build_limb(c, "body", "L_foot").signature)

    def test_limb_dict_round_trip(self):
        a = limb_of(random_chain_urdf(np.random.default_rng(0), 5, (1,)))
        b = specmod.LimbSpec.from_dict(json.loads(json.dumps(a.to_dict())))
        self.assertEqual(a.signature, b.signature)
        q = np.random.default_rng(1).uniform(-1, 1, size=(5, 5))
        np.testing.assert_allclose(a.forward(q, a.nominal), b.forward(q, b.nominal), atol=1e-12)


class TestWriteBack(unittest.TestCase):
    def setUp(self):
        self.r = specmod.from_urdf(robots.quadruped_urdf(), robots.QUADRUPED_MANIFEST)

    def test_identity(self):
        back = self.r.write_back(self.r.params())
        self.assertEqual(urdf.write(back), urdf.write(self.r.urdf))

    def test_new_lengths_reach_the_urdf_with_direction_kept(self):
        new = {"FR": np.array([0.045, 0.13, 0.11])}
        back = self.r.write_back(new)
        self.assertEqual(back.joints["FR_hip_pitch"].xyz, (0.0, -0.045, 0.0))  # right legs keep their sign
        self.assertAlmostEqual(back.joints["FR_foot"].xyz[2], -0.11)
        self.assertEqual(back.joints["FL_foot"].xyz, self.r.urdf.joints["FL_foot"].xyz)
        np.testing.assert_allclose(specmod.params_from_urdf(back, self.r.limbs["FR"]), new["FR"])
        self.assertEqual(self.r.urdf.joints["FR_hip_pitch"].xyz, (0.0, -0.04, 0.0))  # original untouched

    def test_bad_lengths_are_refused(self):
        with self.assertRaises(specmod.SpecError):
            self.r.write_back({"FL": np.array([0.04, 0.12])})
        with self.assertRaises(specmod.SpecError):
            self.r.write_back({"FL": np.array([0.04, -0.12, 0.12])})

    def test_range_check_flags_lengths_the_network_never_saw(self):
        limb = self.r.limbs["FL"]
        self.assertEqual(limb.check_params(limb.nominal * 1.05), [])
        warn = limb.check_params(limb.nominal * np.array([1.0, 1.3, 1.0]))
        self.assertEqual(len(warn), 1)
        self.assertIn("FL_knee", warn[0])


class TestExamplesInSync(unittest.TestCase):
    def test_checked_in_robots_match_the_generators(self):
        with tempfile.TemporaryDirectory() as d:
            for name in robots.write_examples(d):
                with self.subTest(name):
                    with open(os.path.join(d, name)) as a, open(os.path.join(ROOT, "robots", name)) as b:
                        self.assertEqual(a.read(), b.read())


class TestGeneratedNetworks(unittest.TestCase):
    def test_sizes_come_from_the_chain(self):
        limb = limb_of(random_chain_urdf(np.random.default_rng(2), 5))
        net = IKNet(limb, hidden=(16, 16), seed=0)
        self.assertEqual(net.W[0].shape, (3 + limb.n_params, 16))
        self.assertEqual(net.W[-1].shape, (16, 5))
        p = np.zeros((7, 3))
        self.assertEqual(net.predict(p, limb.nominal).shape, (7, 5))
        q = net.predict(p, np.tile(limb.nominal, (7, 1)))
        self.assertTrue(np.all(q >= limb.limits[:, 0]) and np.all(q <= limb.limits[:, 1]))

    def test_backprop_matches_finite_differences_on_a_generated_net(self):
        for dof, pris in ((2, ()), (4, (1,))):
            with self.subTest(dof=dof):
                rng = np.random.default_rng(4)
                limb = limb_of(random_chain_urdf(rng, dof, pris))
                net = IKNet(limb, hidden=(8, 8), seed=1)
                for w in net.W:
                    w += rng.normal(0, 0.1, w.shape)
                p, L, _ = leg.sample_batch(rng, 64, limb)
                loss, grads = net.loss_and_grads(p, L)
                for k, (arr, g) in enumerate(zip(net.params, grads)):
                    idx = tuple(rng.integers(0, s) for s in arr.shape)
                    old, h = arr[idx], 1e-6
                    arr[idx] = old + h
                    hi = net.loss_and_grads(p, L)[0]
                    arr[idx] = old - h
                    lo = net.loss_and_grads(p, L)[0]
                    arr[idx] = old
                    self.assertAlmostEqual(g[idx], (hi - lo) / (2 * h), delta=1e-6 * max(1, abs(g[idx]) * 100))

    def test_sampling_works_for_other_dof(self):
        for dof in (2, 4):
            limb = limb_of(random_chain_urdf(np.random.default_rng(dof), dof))
            p, L, q = leg.sample_batch(np.random.default_rng(0), 100, limb)
            self.assertEqual((p.shape, L.shape, q.shape), ((100, 3), (100, limb.n_params), (100, dof)))
            np.testing.assert_allclose(limb.forward(q, L), p, atol=1e-12)

    def test_short_training_run_reduces_error(self):
        limb = specmod.build_limb(robots.planar_urdf(), "body", "foot")
        net = IKNet(limb, hidden=(32, 32), seed=0)
        rng = np.random.default_rng(0)
        first = last = None
        for step in range(300):
            p, L, _ = leg.sample_batch(rng, 256, limb)
            loss = net.train_step(p, L, 3e-3)
            first = loss if first is None else first
            last = loss
        self.assertLess(last, first * 0.2)


class TestWeightsAlignment(unittest.TestCase):
    def setUp(self):
        self.limb = specmod.build_limb(robots.leg_urdf(), "body", "L_foot", reach=0.24)

    def test_legacy_weights_load_against_a_matching_urdf_and_predict_identically(self):
        plain = IKNet.load(WEIGHTS)
        checked = IKNet.load(WEIGHTS, spec=self.limb)
        p, L, _ = leg.sample_batch(np.random.default_rng(0), 500, plain.cfg)
        np.testing.assert_allclose(checked.predict(p, L), plain.predict(p, L), atol=1e-12)
        self.assertIsInstance(checked.cfg, specmod.LimbSpec)

    def test_legacy_weights_reject_a_different_leg(self):
        m = robots.leg_urdf()
        m.joints["L_knee"].lower = -2.0
        bad = specmod.build_limb(m, "body", "L_foot")
        with self.assertRaises(SpecMismatch) as cm:
            IKNet.load(WEIGHTS, spec=bad)
        self.assertIn("joint limits", str(cm.exception))
        m = robots.leg_urdf()
        m.joints["L_knee"].axis = (1.0, 0.0, 0.0)
        with self.assertRaisesRegex(SpecMismatch, "forward kinematics"):
            IKNet.load(WEIGHTS, spec=specmod.build_limb(m, "body", "L_foot"))
        with self.assertRaisesRegex(SpecMismatch, "2 and 2"):
            IKNet.load(WEIGHTS, spec=specmod.build_limb(robots.planar_urdf(), "body", "foot"))

    def test_new_weights_embed_the_spec_and_check_it_on_load(self):
        net = IKNet(self.limb, hidden=(8,), seed=0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.npz")
            net.save(path)
            back = IKNet.load(path)  # no URDF needed: the file is self-describing
            self.assertEqual(back.cfg.signature, self.limb.signature)
            IKNet.load(path, spec=self.limb)
            # lengths changed in the URDF are fine: they are inputs, not structure
            IKNet.load(path, spec=specmod.build_limb(robots.leg_urdf(lengths=(0.05, 0.13, 0.11)), "body", "L_foot"))
            m = robots.leg_urdf()
            m.joints["L_hip_pitch"].upper = 1.0
            with self.assertRaises(SpecMismatch) as cm:
                IKNet.load(path, spec=specmod.build_limb(m, "body", "L_foot"))
            self.assertIn("limits[1]", str(cm.exception))

    def test_json_export_carries_the_spec(self):
        data = json.loads(IKNet(self.limb, hidden=(8,)).to_json())
        self.assertEqual(data["spec"]["signature"], self.limb.signature)
        self.assertEqual(data["reach"], 0.24)


class NumericSolver:
    """Stand-in for a trained network: exact to 1e-9 and independent of any weights."""

    def __init__(self, limb):
        self.limb = limb

    def predict(self, p, params):
        return self.limb.numeric_ik(p, params, iters=200)


class TestURDFRobot(unittest.TestCase):
    def test_quadruped_agrees_with_the_hard_coded_robot(self):
        r = specmod.from_urdf(robots.quadruped_urdf(), robots.QUADRUPED_MANIFEST)
        new = body.URDFRobot(r, NumericSolver(r.groups["leg"].canonical))
        old = body.Robot(body.AnalyticSolver())
        stance = new.stance()
        np.testing.assert_allclose(np.array([stance[n] for n in new.names]), old.stance(), atol=1e-12)
        q = new.solve(stance)
        np.testing.assert_allclose(np.array([q[n] for n in new.names]), old.solve(old.stance()), atol=1e-6)
        for n, e in new.tracking_error(stance, q).items():
            self.assertLess(e, 1e-8, n)
        # forward kinematics agrees on arbitrary joint values too, including the mirrored legs
        rng = np.random.default_rng(0)
        qq = rng.uniform(leg.JOINT_LIMITS[:, 0], leg.JOINT_LIMITS[:, 1], size=(5, 4, 3))
        feet = new.feet({n: qq[:, i] for i, n in enumerate(new.names)})
        np.testing.assert_allclose(np.stack([feet[n] for n in new.names], axis=1), old.feet(qq), atol=1e-12)

    def test_hexapod_and_per_limb_lengths(self):
        r = specmod.from_urdf(robots.hexapod_urdf(), robots.HEXAPOD_MANIFEST)
        rob = body.URDFRobot(r, NumericSolver(r.groups["leg"].canonical))
        self.assertEqual(len(rob.names), 6)
        rob.params["MR"] = rob.params["MR"] * np.array([1.0, 1.05, 0.95])  # one leg differs
        target = rob.stance()
        err = rob.tracking_error(target)
        self.assertTrue(all(e < 1e-8 for e in err.values()))
        self.assertGreater(np.linalg.norm(rob.solve(target)["MR"] - rob.solve(target)["ML"]), 0.0)

    def test_needs_a_solver_per_group(self):
        m = robots.quadruped_urdf()
        m.joints["RL_knee"].lower = -2.0
        r = specmod.from_urdf(m)
        with self.assertRaises(ValueError):
            body.URDFRobot(r, NumericSolver(next(iter(r.groups.values())).canonical))


if __name__ == "__main__":
    unittest.main()
