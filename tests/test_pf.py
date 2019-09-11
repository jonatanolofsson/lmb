"""Test PF implementation."""

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pylmb as lmb

np.random.seed(0)


class TestPF(unittest.TestCase):
    """Testcases for PF Target."""

    def setUp(self):
        """Set up."""
        self.model = lmb.models.ConstantVelocityModel(0.1, 1)
        self.x0 = np.array([0.0] * 4)
        dT = 1
        Q = np.array([[dT ** 3 / 3, 0,           dT ** 2 / 2, 0],
                      [0,           dT ** 3 / 3, 0,           dT ** 2 / 2],
                      [dT ** 2 / 2, 0,           dT,          0],
                      [0,           dT ** 2 / 2, 0,           dT]])
        self.P0 = Q
        self.filter = MagicMock()
        self.filter.kappa = MagicMock(return_value=1)
        self.pf = lmb.pf.PF.from_gaussian(self.x0, self.P0, 1000000)

    def test_gaussian_mean(self):
        """Test Gaussian mean from init."""
        self.assertAlmostEqual(np.linalg.norm(self.pf.mean() - self.x0), 0.0, 1)  # noqa

    def test_gaussian_cov(self):
        """Test Gaussian covariance from init."""
        self.assertAlmostEqual(np.linalg.norm(self.pf.cov() - self.P0), 0.0, 1)  # noqa

    def test_predict(self):
        """Predict step."""
        dT = 1
        params = MagicMock()
        params.N_max = 100000
        self.pf.predict(params, self.model, dT)
        self.assertAlmostEqual(np.linalg.norm(self.pf.mean()[0] - self.x0[0] - self.x0[2] * dT), 0.0, 1)  # noqa

    def test_correct(self):
        """Correction step."""
        z = np.array([2.0] * 2)
        R = np.eye(2)
        params = MagicMock()
        params.kappa = MagicMock(return_value=1)
        r = lmb.GaussianReport(z, R)
        r.sensor = lmb.sensors.EyeOfMordor()
        self.pf.correct(params, r)
        x = self.pf.mean()
        self.assertAlmostEqual(x[0], 0.5, 1)
        self.assertAlmostEqual(x[1], 0.5, 1)
        self.assertAlmostEqual(x[2], 0.75, 1)
        self.assertAlmostEqual(x[3], 0.75, 1)
