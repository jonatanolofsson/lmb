"""File for sensor-related stuff."""

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

from .utils import LARGE, within
from .models import position_measurement


class EyeOfMordor:
    """Ideal sensor that sees all."""

    def __init__(self):
        """Init."""
        self.model = position_measurement

    def bbox(self):
        """Return FOV bbox."""
        return (-LARGE, LARGE, -LARGE, LARGE)

    def in_fov(self, state):
        """Return nll prob of detection, given fov."""
        return True

    def pD(self, states):
        """Probability of detection for states x."""
        return np.ones(len(states))


class Satellite:
    """Satellite sensor with field-of-view."""

    def __init__(self, fov):
        """Init."""
        self.model = position_measurement
        self.fov = fov

    def bbox(self):
        """Return FOV bbox."""
        return self.fov

    def in_fov(self, state):
        """Return nll prob of detection, given fov."""
        return within(state, self.fov)

    def pD(self, states):
        """Probability of detection for states x."""
        return np.array([self.in_fov(x) for x in states])
