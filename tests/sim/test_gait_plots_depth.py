"""toward_camera: the depth number that orders legs against the body in the isometric animations."""
import unittest

import numpy as np

from mojito.sim import gait_plots


class TestTowardCamera(unittest.TestCase):
    def test_matches_camera_position_of_matplotlib_view(self):
        # Matplotlib's camera sits at (cos e cos a, cos e sin a, sin e) from the centre: that point is nearest.
        elev, azim = 35.264, -45.0
        camera = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                           np.cos(np.radians(elev)) * np.sin(np.radians(azim)), np.sin(np.radians(elev))])
        self.assertAlmostEqual(float(gait_plots.toward_camera(camera, elev, azim)), 1.0)
        self.assertAlmostEqual(float(gait_plots.toward_camera(-camera, elev, azim)), -1.0)

    def test_shoulder_behind_body_is_farther(self):
        # azim -45: the camera is toward +x, -y. A shoulder at -x, +y is behind the body centre.
        behind = gait_plots.toward_camera([-0.1, 0.05, 0.0], 35.264, -45)
        centre = gait_plots.toward_camera([0.0, 0.0, 0.0], 35.264, -45)
        front = gait_plots.toward_camera([0.1, -0.05, 0.0], 35.264, -45)
        self.assertLess(behind, centre)
        self.assertGreater(front, centre)


if __name__ == "__main__":
    unittest.main()
