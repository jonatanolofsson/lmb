"""Helper functions for MHT plots."""

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

import matplotlib.colors
from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

from .utils import cov_ellipse

CMAP = matplotlib.colors.ListedColormap(RandomState(0).rand(256*256, 3))


def plot_trace(t, c=0, covellipse=True, max_back=None, r_values=False, track_id=False, velocity=False, trace=True, **kwargs):
    """Plot single trace."""
    max_back = max_back or 0
    xs, ys, vxs, vys = [], [], [], []
    for ty, r, x, P in t.history[-max_back:]:
        state = x.tolist()
        xs.append(state[0])
        ys.append(state[1])
        vxs.append(state[2])
        vys.append(state[3])
    if covellipse:
        ca = plot_cov_ellipse(P[0:2, 0:2], state[0:2], 4)
        ce = plot_cov_ellipse(P[0:2, 0:2], state[0:2], 4)
        ca.set_alpha(0.2)
        ca.set_facecolor(CMAP(c))
        ce.set_facecolor('none')
        ce.set_edgecolor(CMAP(c))
        ce.set_linewidth(4)        
    if trace:
        plt.plot(xs, ys, color=CMAP(c), **kwargs)
    if r_values:    
        plt.text(t.pdf.mean()[0]+20,t.pdf.mean()[1], '{0:.2f}'.format(t.r), color=CMAP(c), fontsize=16)
    if track_id:
        plt.text(t.pdf.mean()[0]+20,t.pdf.mean()[1]+20, str(t.id), color=CMAP(c), )
    if velocity:
        v = np.array([vxs[-1], vys[-1]])
        va = np.sqrt(vxs[-1]*vxs[-1] + vys[-1]*vys[-1])
        if va > 15:
            v *= 15 / va
        plt.plot([xs[-1],xs[-1]+v[0]], [ys[-1],ys[-1]+v[1]], color=CMAP(c), linewidth=3)

def plot_traces(targets, cseed=0, covellipse=True, max_back=None, **kwargs):
    """Plot all targets' traces."""
    for t in targets:
        plot_trace(t, t.id + cseed, covellipse, max_back, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """Plot confidence ellipse."""
    r1, r2, theta = cov_ellipse(cov, nstd)
    ellip = Ellipse(xy=pos, width=2*r1, height=2*r2, angle=theta, **kwargs)

    plt.gca().add_artist(ellip)
    return ellip


def plot_scan(scan, covellipse=True, **kwargs):
    """Plot reports from scan."""
    options = {
        'marker': '+',
        'color': 'r',
        'linestyle': 'None'
    }
    options.update(kwargs)
    plt.plot([float(r.z[0]) for r in scan.reports],
             [float(r.z[1]) for r in scan.reports], **options)
    if covellipse:
        for r in scan.reports:
            ca = plot_cov_ellipse(r.R[0:2, 0:2], r.z[0:2])
            ca.set_alpha(0.1)
            ca.set_facecolor(options['color'])


def plot_bbox(obj, cseed=0, **kwargs):
    """Plot bounding box."""
    id_ = getattr(obj, 'id', 0)
    options = {
        'alpha': 0.3,
        'color': CMAP(id_ + cseed)
    }
    options.update(kwargs)
    bbox = obj.bbox()
    plt.gca().add_patch(Rectangle(
        (bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2],
        **options))
