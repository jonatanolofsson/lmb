"""Motion and measurement models."""

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

import numpy as np
from scipy.stats import multivariate_normal


def sample_normal(Q, N):
    """Generate random samples and their weights."""
    var = multivariate_normal(np.zeros(Q.shape[0]), Q)
    s = var.rvs(N)
    return s


class ConstantVelocityModel:
    """Constant velocity motion model."""

    def __init__(self, q, pS):
        """Init."""
        self.q = q
        self.pS = pS

    def __call__(self, pdf, dT):
        """Step model."""
        F = np.array([[1, 0, dT, 0],
                       [0, 1, 0, dT],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        Q = np.array([[dT ** 3 / 3, 0,           dT ** 2 / 2, 0],
                      [0,           dT ** 3 / 3, 0,           dT ** 2 / 2],
                      [dT ** 2 / 2, 0,           dT,          0],
                      [0,           dT ** 2 / 2, 0,           dT]]) * self.q

        e = sample_normal(Q, len(pdf.x))
        pdf.x = (F @ pdf.x.T).T + e
        pdf.w *= self.pS


def position_measurement(x):
    """Velocity measurement model."""
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    return (H @ x.T).T


def velocity_measurement(x):
    """Velocity measurement model."""
    H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    return (H @ x.T).T

class UniformClutter:
    """Uniform clutter model."""

    def __init__(self, intensity):
        """Init."""
        self.intensity = intensity

    def __call__(self, z):
        """Evaluate at z."""
        return self.intensity
