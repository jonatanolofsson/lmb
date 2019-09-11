"""Particle tracker for LMB tracker."""

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
from .utils import gaussian_bbox, gaussian_aa_bbox


class PF:
    """Particle tracker implementation."""

    def __init__(self, prior=None):
        """Init."""
        if prior:
            self.w, self.x = prior
        else:
            self.clear()
        self.normalize()

    def from_gaussian(z, R, N):
        """Init new PF from gaussian prior."""
        var = multivariate_normal(z, R)
        s = var.rvs(N)
        w = np.full((N,), 1/N)

        return PF((w, s))

    def join(weights, pdfs):
        """Join multiple PFs into one."""
        w = [w * pdf.w for w, pdf in zip(weights, pdfs)
             if w > 1e-6 and len(pdf.x) > 0]
        if len(w) > 0:
            w = np.concatenate(w)
            x = [pdf.x for w, pdf in zip(weights, pdfs)
                 if w > 1e-6 and len(pdf.x) > 0]
            x = np.concatenate(x) if len(x) else np.empty(0)
        else:
            w, x = np.empty(0), np.empty(0)
        self = PF((w, x))
        return self

    def predict(self, params, model, dT):
        """Prediction step."""
        self.resample(params)
        model(self, dT)
        self.eta = self.w.sum()
        self.normalize()

    def correct(self, params, r):
        """Correction step."""
        self.w *= self.psi(params, r)
        self.eta = self.w.sum()
        if self.eta > 1e-10:
            self.w /= self.eta
        else:
            self.clear()
        return self

    def psi(self, params, r):
        """Corrected (non-normalized) weight."""
        if r is not None:
            # pD(x,l) * pG(x,l) * g(r|x) / k(z)
                # * tracker.pG \  # FIXME
            return r.sensor.pD(self.x) \
                * r.likelihood(self.x) \
                / params.kappa(r)
        else:
            sensor = params
            # 1 - pD(x,l) * pG(x,l)
            return 1 - sensor.pD(self.x)  # * self.tracker.pG

    def clear(self):
        """Clear filter."""
        self.w, self.x = (np.empty(0), np.empty(0))

    def normalize(self):
        """Re-normalize the PDF."""
        self.w /= self.w.sum()

    def neff(self):
        """Return effective number of particles."""
        return 1.0 / sum(w**2 for w in self.w)

    def resample(self, params=None, N=None, neff=None):
        """Resample algorithm."""
        N0 = params.N_max if params else len(self.x)
        N = N or N0
        if N == 0:
            return
        if N == N0 and neff is not None and neff < self.neff()/N:
            return
        self.normalize()
        C = np.cumsum(self.w)
        u0 = np.random.uniform(0, 1/N)
        m = 0
        i = 0
        x = np.empty((N, self.x.shape[1]))
        for u in np.arange(u0, 1+u0, 1/N):
            while C[m] < u:
                m += 1
            x[i] = self.x[m]
            i += 1
        N = len(x)
        self.x = x
        self.w = np.full((N,), 1/N)

    def mean(self):
        """Calculate PDF mean."""
        return np.average(self.x, 0, self.w)

    def cov(self):
        """Calculate variance."""
        return np.cov(self.x, rowvar=False, aweights=self.w)

    def bbox(self, nstd=2):
        """Get pdf bbox."""
        return gaussian_bbox(self.mean()[0:2], self.cov()[0:2, 0:2])

    def aa_bbox(self, nstd=2):
        """Get axis-aligned pdf bbox."""
        return gaussian_aa_bbox(self.mean()[0:2], self.cov()[0:2, 0:2])

    def __bool__(self):
        """Boolean operator."""
        return len(self.w) > 0
