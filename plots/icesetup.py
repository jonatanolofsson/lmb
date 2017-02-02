"""Create crosstrack.png plot."""

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
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
import lmb

np.random.seed(1)

simlen = 50
N_ice = 100
xmin, xmax = 0, 1000
mid = np.array([[(xmax-xmin) / 2]] * 2)
Pz = np.diag([0.01] * 2)
Pz_true = np.diag([0.01] * 2)

class df:
    pass


def rot(x):
    """Rotate vectors 90 deg ccw."""
    return np.array([[0, -1], [1, 0]]) @ x


def init_tracker():
    """Init tracker."""
    params = lmb.Parameters()
    params.N_max = 50000
    params.lambdaB = 0.1
    params.kappa = lmb.models.UniformClutter(0.0001)
    params.init_target = lmb.DefaultTargetInit(0.01, 1, 1)
    params.r_lim = 0.02
    params.nstd = 1.9
    return lmb.LMB(params)


def init_ice():
    """Init ice objects."""
    res = df()
    v = 1
    pv = np.diag([0.05] * 2)
    res.positions = np.random.uniform(xmin, xmax, (2, N_ice))
    mid_vecs = (mid - res.positions)
    mid_vecs /= np.sqrt((mid_vecs ** 2).sum(axis=0))
    res.velocities = rot(mid_vecs * v)
    res.velocities += np.random.multivariate_normal(np.array([0.0]*2), pv,
                                                    size=N_ice).T
    return res


def init_planes():
    """Init planes."""
    res = df()
    res.positions = np.array([[500], [0]])
    res.directions = np.array([[0], [5]])
    res.amplitudes = np.array([[300]])
    res.omegas = np.array([[0.05]])
    res.phases = np.array([[0]])
    res.fovs_l = np.array([[-60]])
    res.fovs_r = np.array([[60]])
    res.fovs_b = np.array([[-20]])
    res.fovs_f = np.array([[60]])
    return res


def update_ice(targets):
    """Move icebergs."""
    a = 0.1
    pa = np.diag([0.001] * 2)
    mid_vecs = (mid - targets.positions)
    mid_vecs /= np.sqrt((mid_vecs ** 2).sum(axis=0))
    accelerations = mid_vecs * a
    accelerations += np.random.multivariate_normal(
        np.array([0.0]*2), pa, size=accelerations.shape[1]).T
    targets.positions += targets.velocities
    targets.velocities += accelerations


def plane_states(planes, t):
    """Update planes."""
    res = df()
    orthodir = rot(planes.directions)
    orthodir = orthodir / np.sqrt((orthodir ** 2).sum(axis=0))

    res.positions = planes.positions + planes.directions * t + \
        planes.amplitudes * np.sin(planes.omegas * t + planes.phases) \
        * orthodir

    res.velocities = planes.directions + \
        planes.amplitudes * np.cos(planes.omegas * t + planes.phases) \
        * orthodir * planes.omegas
    res.front = res.velocities / np.sqrt((res.velocities ** 2).sum(axis=0))
    res.left = rot(res.front)

    res.fovs_fl = res.positions + res.front * planes.fovs_f - res.left * planes.fovs_l
    res.fovs_bl = res.positions + res.front * planes.fovs_b - res.left * planes.fovs_l
    res.fovs_br = res.positions + res.front * planes.fovs_b - res.left * planes.fovs_r
    res.fovs_fr = res.positions + res.front * planes.fovs_f - res.left * planes.fovs_r

    return res


def look_for_ice(routes, planes, icebergs):
    """Retrieve, for each plane, the visible ice."""
    res = []
    for p in range(planes.positions.shape[1]):
        fov = Polygon([planes.fovs_fl[:, p], planes.fovs_bl[:, p],
                       planes.fovs_br[:, p], planes.fovs_fr[:, p]])
        res.append([i for i in range(icebergs.positions.shape[1])
            if fov.contains(Point(icebergs.positions[0, i], icebergs.positions[1, i]))])
    return res


def ice_to_scans(planes, icebergs, findings):
    """Create a scan for each plane."""
    return [lmb.Scan(lmb.sensors.SquareSensor(
        [planes.fovs_fl[:, p], planes.fovs_bl[:, p],
         planes.fovs_br[:, p], planes.fovs_fr[:, p]]),
        [lmb.GaussianReport(np.random.multivariate_normal(
            icebergs.positions[:, i], Pz_true),
                    Pz, i) for i in f])
        for p, f in enumerate(findings)]


def draw():
    """Create plot."""
    plt.figure(figsize=(20, 4 * simlen // 5))
    tracker = init_tracker()
    icebergs = init_ice()
    routes = init_planes()
    plane_history = []

    for k in range(simlen):
        print(k)
        plt.subplot(simlen / 5, 5, k + 1)
        if k > 0:
            update_ice(icebergs)
            tracker.predict()

        plt.quiver(icebergs.positions[0, :], icebergs.positions[1, :],
                   icebergs.velocities[0, :], icebergs.velocities[1, :])

        planes = plane_states(routes, k)
        plane_history.append(planes)

        plt.quiver(planes.positions[0, :], planes.positions[1, :],
                   planes.velocities[0, :], planes.velocities[1, :], color='r')

        findings = look_for_ice(routes, planes, icebergs)
        scans = ice_to_scans(planes, icebergs, findings)
        for scan in scans:
            tracker.register_scan(scan)

        for i in range(planes.positions.shape[1]):
            plt.gca().add_patch(
                patches.Polygon(
                    [planes.fovs_fl[:, i], planes.fovs_bl[:, i],
                     planes.fovs_br[:, i], planes.fovs_fr[:, i]],
                    closed=True, fill=True, color='b', alpha=0.4))
            hist_x = [h.positions[0, i] for h in plane_history]
            hist_y = [h.positions[1, i] for h in plane_history]
            plt.plot(hist_x, hist_y, 'r')

            plt.quiver(icebergs.positions[0, findings[i]],
                       icebergs.positions[1, findings[i]],
                       icebergs.velocities[0, findings[i]],
                       icebergs.velocities[1, findings[i]],
                       color='g')

        tracked_targets = tracker.query_targets()
        lmb.plot.plot_traces(tracked_targets, covellipse=True)

        plt.xlim([xmin, xmax])
        plt.ylim([xmin, xmax])


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    draw()
    if args.show:
        plt.show()
    else:
        plt.gcf().savefig(os.path.splitext(os.path.basename(__file__))[0],
                          bbox_inches='tight')


if __name__ == '__main__':
    main(*sys.argv[1:])
