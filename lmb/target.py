"""Class to represent targets."""

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
from copy import deepcopy
from .utils import nll


class Target:
    """Class to represent a single target."""

    def __init__(self, id, model, r=0, pdf=None):
        """Init."""
        self.id = id
        self.model = model
        self.r = r
        self.pdf = pdf
        self.assignments = {}
        self.history = []
        self.history.append(('i', self.pdf.mean(), self.pdf.cov()))

    def correct(self, weights):
        """Create new object to represent this target in the next timestep."""
        self.r = weights.sum()
        if self.r > 1e-9:
            self.pdf = type(self.pdf).join(weights / self.r, self.assignments)
            self.history.append(('c', self.pdf.mean(), self.pdf.cov()))
        else:
            self.r = 0
            self.pdf.clear()
            self.history.append(('c', False, False))

    def predict(self, params, dT):
        """Move to next time step."""
        self.pdf.predict(params, self.model, dT)
        self.r *= self.pdf.eta
        if self.r > 1e-6:
            self.history.append(('p', self.pdf.mean(), self.pdf.cov()))
        else:
            self.history.append(('p', False, False))
            self.pdf.clear()

    def match(self, params, sensor, reports):
        """Create new filters from hypothetical assignments."""
        self.assignments = [deepcopy(self.pdf).correct(params, r)
                            for r in reports]
        etas = [nll(self.r * pdf.eta) for pdf in self.assignments]
        self.assignments.append(deepcopy(self.pdf).correct(sensor, None))
        return etas

    def missed(self):
        """NLL of target being missed."""
        return nll(self.r * self.assignments[-1].eta)

    def false(self):
        """NLL of target being false."""
        return nll(1 - self.r)

    def bbox(self, nstd=2):
        """Return target bounding box."""
        return self.pdf.bbox(nstd)

    def aa_bbox(self, nstd=2):
        """Return axis-aligned target bounding box."""
        return self.pdf.aa_bbox(nstd)

    def normalize(self, w):
        """Normalize target pdf and weighting."""
        self.r /= w
        if self.r > 1.0:
            self.r = 1.0
        self.pdf.normalize()

    def __repr__(self):
        """String representation of object."""
        return "T({} / {})".format(self.id, self.r)
