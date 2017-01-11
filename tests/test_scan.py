"""Test scan implementation."""

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
import numpy as np
import os
import sys
from math import pi, exp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lmb

np.random.seed(0)


class TestReport(unittest.TestCase):
    """Testcases for reports."""

    def setUp(self):
        """Set up."""
        self.model = lmb.models.ConstantVelocityModel(0.1, 1)
        self.x = np.array([0.0] * 2)
        self.R = np.eye(2)
        self.r = lmb.scan.GaussianReport(self.x, self.R)
        self.r.sensor = lmb.sensors.EyeOfMordor()

    def test_gaussian_mean(self):
        """Test Gaussian mean from init."""
        self.assertAlmostEqual(np.linalg.norm(self.r.z - self.x), 0.0)  # noqa

    def test_gaussian_cov(self):
        """Test Gaussian covariance from init."""
        self.assertAlmostEqual(np.linalg.norm(self.r.R - self.R), 0.0)  # noqa

    def test_likelihood(self):
        """Test likelihood function."""
        x = np.array([[0.0] * 4, [0.5] * 4, [1.0] * 4])
        res = self.r.likelihood(x)
        self.assertEqual(len(res), 3)
        self.assertAlmostEqual(res[0], 1 / (2 * pi))
        self.assertAlmostEqual(res[1], exp(-(0.5 ** 2)) / (2 * pi))
        self.assertAlmostEqual(res[2], exp(-(1.0 ** 2)) / (2 * pi))
