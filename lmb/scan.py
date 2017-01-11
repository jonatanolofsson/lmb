"""Scan and Report class."""

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
from .utils import gaussian_bbox
from scipy.stats import multivariate_normal


class GaussianReport:
    """Class for containing reports."""

    def __init__(self, z, R, source=None, tpos=None):
        """Init."""
        self.z = z
        self.R = R
        self.sensor = None
        self.source = source
        self.tpos = tpos
        self._bbox = gaussian_bbox(self.z[0:2], self.R[0:2, 0:2], 2)
        self.ruk = 0
        self.rB = 0

    def likelihood(self, x):
        """Calculate the likelihood of x generating report."""
        return multivariate_normal.pdf(self.sensor.model(x), self.z, self.R)

    def bbox(self):
        """Return report bbox."""
        return self._bbox

    def __repr__(self):
        """Return string representation of reports."""
        return "R({}, R)".format(self.z.T)


class Scan:
    """Report container class."""

    def __init__(self, sensor, reports):
        """Init."""
        self.sensor = sensor
        self.reports = reports
        for r in self.reports:
            r.sensor = sensor

    def __repr__(self):
        """Return a string representation of the scan."""
        return "Scan: {}".format(str(self.reports))
