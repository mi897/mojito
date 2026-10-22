"""The drive-mode input mapping (site/assets/drive.js) that the live pages share. Run with:  python -m unittest discover tests"""
import json
import os
import shutil
import subprocess
import unittest

NODE = shutil.which("node")
HERE = os.path.dirname(os.path.abspath(__file__))


def run_js(expression):
    script = "const D=require('../../site/assets/drive.js');console.log(JSON.stringify(%s));" % expression
    return json.loads(subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True).stdout)


@unittest.skipIf(NODE is None, "node is not installed")
class TestDriveMapping(unittest.TestCase):
    def test_keys_give_full_stick(self):
        self.assertEqual(run_js("D.fractions(new Set(['fwd','turnl']),null)"), [1, 0, 1])
        back_right = run_js("D.fractions(new Set(['back','right','turnr']),null)")  # back and right together are on the unit circle
        self.assertAlmostEqual(back_right[0], -(0.5 ** 0.5))
        self.assertAlmostEqual(back_right[1], -(0.5 ** 0.5))
        self.assertEqual(back_right[2], -1)

    def test_opposite_keys_cancel(self):
        self.assertEqual(run_js("D.fractions(new Set(['fwd','back','left','right']),null)"), [0, 0, 0])

    def test_diagonal_is_no_faster_than_straight(self):
        f = run_js("D.fractions(new Set(['fwd','left']),null)")
        self.assertAlmostEqual(f[0] ** 2 + f[1] ** 2, 1.0)
        self.assertAlmostEqual(f[0], f[1])

    def test_stick_dead_zone_and_rescaling(self):
        dead = run_js("D.DEAD")
        self.assertEqual(run_js("D.stick(%f)" % (dead * 0.99)), 0)
        self.assertAlmostEqual(run_js("D.stick(1)"), 1.0)
        self.assertAlmostEqual(run_js("D.stick(-1)"), -1.0)
        self.assertAlmostEqual(run_js("D.stick(%f)" % ((1 + dead) / 2)), 0.5)  # halfway between the dead zone and full

    def test_pad_axes_signs(self):
        # standard layout: stick up is y = -1 (forward), stick left is x = -1 (left, and turn left on the right stick)
        self.assertEqual(run_js("D.fractions(new Set(),[0,-1,0,0])"), [1, 0, 0])
        self.assertEqual(run_js("D.fractions(new Set(),[-1,0,0,0])"), [0, 1, 0])
        self.assertEqual(run_js("D.fractions(new Set(),[0,0,-1,0])"), [0, 0, 1])

    def test_a_moved_stick_beats_keys_but_a_resting_one_does_not(self):
        self.assertEqual(run_js("D.fractions(new Set(['fwd']),[0.05,0.05,0,0])"), [1, 0, 0])
        self.assertEqual(run_js("D.fractions(new Set(['fwd']),[0,1,0,0])"), [-1, 0, 0])

    def test_scale_by_preset_limits(self):
        self.assertEqual(run_js("D.scale([1,-1,0.5],{vx:0.25,vy:0.12,yaw_rate:0.8})"), [0.25, -0.12, 0.4])

    def test_detect_preset(self):
        presets = {"trot": {"height": 0.16, "period": 1.7}, "crawl": {"height": 0.2125, "period": 2.0}}
        self.assertEqual(run_js("D.detectPreset({height:0.2125,period:2.0},%s)" % json.dumps(presets)), "crawl")
        self.assertIsNone(run_js("D.detectPreset({height:0.3,period:1.0},%s)" % json.dumps(presets)))


if __name__ == "__main__":
    unittest.main()
