import unittest

import numpy as np

from mojito.geometry.placeholders import SIM_QUADRUPED
from mojito.robot import budget


class TestBudget(unittest.TestCase):
    def test_second_moment_against_a_grid_sum(self):
        # independent of the closed form: integrate y^2 over the section on a fine grid
        tube = budget.Tube(depth=0.020, width=0.014, wall=0.002)
        n = 2000
        y = (np.arange(n) + 0.5) / n * tube.depth - tube.depth / 2
        x = (np.arange(n) + 0.5) / n * tube.width - tube.width / 2
        xx, yy = np.meshgrid(x, y)
        inside = (np.abs(xx) < tube.width / 2 - tube.wall) & (np.abs(yy) < tube.depth / 2 - tube.wall)
        cell = (tube.width / n) * (tube.depth / n)
        self.assertAlmostEqual(np.sum((yy**2) * ~inside) * cell / tube.second_moment, 1.0, delta=2e-3)  # grid edge error
        self.assertAlmostEqual(np.sum(~inside) * cell / tube.area, 1.0, delta=2e-3)

    def test_tip_deflection_against_double_integration(self):
        # cantilever with a tip load F: integrate the curvature M(x)/EI twice
        tube, mat, length, torque = budget.Tube(), budget.PLA, 0.15, 1.5
        force = torque / length
        x = np.linspace(0.0, length, 20001)
        curvature = force * (length - x) / (mat.youngs_modulus * tube.second_moment)
        slope = np.concatenate([[0], np.cumsum((curvature[1:] + curvature[:-1]) / 2 * np.diff(x))])
        sag = np.sum((slope[1:] + slope[:-1]) / 2 * np.diff(x))
        self.assertAlmostEqual(sag / budget.tip_deflection(torque, length, tube, mat), 1.0, places=4)

    def test_bending_stress_scales_with_moment_and_depth(self):
        tube = budget.Tube()
        self.assertAlmostEqual(budget.bending_stress(2.0, tube) / budget.bending_stress(1.0, tube), 2.0)
        deeper = budget.Tube(depth=0.030, width=0.014, wall=0.002)
        self.assertLess(budget.bending_stress(1.0, deeper), budget.bending_stress(1.0, tube))

    def test_totals_conserve_mass(self):
        parts = budget.default_parts(SIM_QUADRUPED)
        total = budget.totals(parts)
        self.assertAlmostEqual(total["total"], sum(p.mass for p in parts))
        for key in ("by_group", "by_source", "by_role"):
            self.assertAlmostEqual(sum(total[key].values()), total["total"])
        roles = budget.role_masses(parts)
        self.assertAlmostEqual(roles["body"] + budget.LEGS * sum(roles[r] for r in ("hip", "thigh", "shank", "foot")), total["total"])

    def test_listed_servo_and_battery_masses(self):
        parts = {p.name: p for p in budget.default_parts(SIM_QUADRUPED)}
        servos = sum(p.mass for p in parts.values() if p.group == "servo")
        self.assertAlmostEqual(servos, 12 * 0.060)
        self.assertAlmostEqual(parts["battery, 2S 1500 mAh"].mass, 0.076)

    def test_longer_legs_weigh_more_and_pla_is_lighter_than_petg(self):
        short = budget.totals(budget.default_parts(SIM_QUADRUPED))["total"]
        import dataclasses
        longer = dataclasses.replace(SIM_QUADRUPED, upper_length=0.2, lower_length=0.2)
        self.assertGreater(budget.totals(budget.default_parts(longer))["total"], short)
        self.assertLess(short, budget.totals(budget.default_parts(SIM_QUADRUPED, material=budget.PETG))["total"])

    def test_leg_scale_to_fit(self):
        self.assertAlmostEqual(budget.leg_scale_to_fit(2.0, 1.8), 0.9)


if __name__ == "__main__":
    unittest.main()
