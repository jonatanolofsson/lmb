"""Create icesetup.png plot."""

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
from math import pi
import pickle

sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
import lmb
from lmb.utils import normalize, rotmat

np.random.seed(1)

N_ice = 20
xmin, xmax = 0, 1000
mid = np.array([[(xmax-xmin) / 2]] * 2)
Pz_true = np.diag([0.005] * 2)
PLOTS_WIDE = 3
PLOTS_HIGH = 3
PLOT_EVERY = 15
simlen = PLOTS_WIDE * PLOTS_HIGH * PLOT_EVERY
plot_rlim = 0.7


class df():
    pass


def rot(x):
    """Rotate vectors 90 deg ccw."""
    return np.array([[0, -1], [1, 0]]) @ x


def init_tracker():
    """Init tracker."""
    params = lmb.Parameters()
    params.N_max = 500  # FIXME
    params.lambdaB = lambda r: r.sensor.lambdaB
    params.kappa = lmb.models.UniformClutter(0.0001)
    params.init_target = lmb.DefaultTargetInit(q=0.01, pv=1, pS=0.999)
    params.r_lim = 0.02
    params.nstd = 1.9
    return lmb.LMB(params)


def init_ice():
    """Init ice objects."""
    res = df()
    v = 0.8
    pv = np.diag([0.2] * 2)
    res.positions = np.random.uniform(xmin, xmax, (2, N_ice))
    mid_vecs = (mid - res.positions)
    mid_vecs /= np.sqrt((mid_vecs ** 2).sum(axis=0))
    res.velocities = rot(mid_vecs * v)
    res.velocities += np.random.multivariate_normal(np.array([0.0]*2), pv,
                                                    size=N_ice).T
    return res


def init_planes():
    """Init planes."""
    res = []

    plane = df()
    plane.i = len(res)
    plane.model = sin_model
    plane.initial_position = np.array([[500], [0]])
    plane.direction = np.array([[0], [6]])
    plane.amplitude = 300
    plane.omega = 0.08
    plane.phase = 0
    plane.fov_l = -60
    plane.fov_r = 60
    plane.fov_b = -20
    plane.fov_f = 60
    plane.history = []
    plane.active = lambda k: True
    plane.plot = True
    plane.Pz = np.diag([0.1] * 2)
    plane.p_detect = 0.99
    plane.lambdaB = 0.1
    res.append(plane)

    plane = df()
    plane.i = len(res)
    plane.model = circ_model
    plane.initial_position = np.array([[700], [500]])
    plane.midpoint = np.array([[500], [500]])
    plane.omega = 9 * pi / 180
    plane.fov_l = -60
    plane.fov_r = 60
    plane.fov_b = -20
    plane.fov_f = 60
    plane.history = []
    plane.active = lambda k: True
    plane.plot = True
    plane.Pz = np.diag([0.1] * 2)
    plane.p_detect = 0.99
    plane.lambdaB = 0.1
    res.append(plane)

    plane = df()
    plane.i = len(res)
    plane.model = lambda p, k: None
    plane.position = np.array([[-500], [-500]])
    plane.velocity = np.array([[1], [0]])
    plane.fov_l = -10000
    plane.fov_r = 10000
    plane.fov_b = -10000
    plane.fov_f = 10000
    plane.history = []
    plane.active = lambda k: ((k % 75) == 0)
    plane.plot = False
    plane.Pz = np.diag([5] * 2)
    plane.p_detect = 0.9
    plane.lambdaB = 15
    res.append(plane)

    return res


def update_ice(targets):
    """Move icebergs."""
    a = 0.01
    pa = np.diag([0.001] * 2)
    mid_vecs = (mid - targets.positions)
    mid_vecs /= np.sqrt((mid_vecs ** 2).sum(axis=0))
    accelerations = mid_vecs * a
    # accelerations += np.random.multivariate_normal(
        # np.array([0.0]*2), pa, size=accelerations.shape[1]).T
    targets.positions += targets.velocities
    targets.velocities += accelerations


def sin_model(plane, t):
    """Sine motion model for planes."""
    orthodir = normalize(rot(plane.direction))

    plane.position = plane.initial_position + plane.direction * t + \
        plane.amplitude * np.sin(plane.omega * t + plane.phase) \
        * orthodir

    plane.velocity = plane.direction + \
        plane.amplitude * np.cos(plane.omega * t + plane.phase) \
        * orthodir * plane.omega


def circ_model(plane, t):
    """Circular motion model."""
    pos = rotmat(plane.omega * t) @ (plane.initial_position - plane.midpoint)

    plane.position = pos + plane.midpoint
    plane.velocity = plane.omega * np.linalg.norm(pos) * normalize(rot(pos))


def update_planes(planes, k):
    """Update planes."""
    for p in planes:
        p.model(p, k)
        p.front = normalize(p.velocity)
        p.left = rot(p.front)

        p.fov = (p.position + p.front * p.fov_f - p.left * p.fov_l).T.tolist() + \
                (p.position + p.front * p.fov_b - p.left * p.fov_l).T.tolist() + \
                (p.position + p.front * p.fov_b - p.left * p.fov_r).T.tolist() + \
                (p.position + p.front * p.fov_f - p.left * p.fov_r).T.tolist()

        p.history.append(p.position)


def look_for_ice(planes, icebergs, k):
    """Retrieve, for each plane, the visible ice."""
    for p in planes:
        if p.active(k):
            fov = Polygon(p.fov)
            p.findings = [i for i in range(icebergs.positions.shape[1])
                if fov.contains(Point(icebergs.positions[0, i], icebergs.positions[1, i]))]
            p.scan = lmb.Scan(
                lmb.sensors.SquareSensor(p.fov, p.p_detect, p.lambdaB),
                [lmb.GaussianReport(np.random.multivariate_normal(
                    icebergs.positions[:, i], Pz_true), p.Pz, i)
                    for i in p.findings])


def draw():
    """Create plot."""

    def mkplot(observed=False):
        plt.plot([500], [500], marker='h', markersize=40, color='c')
        plt.quiver(
            [p.position[0, 0] for p in planes],
            [p.position[1, 0] for p in planes],
            [p.velocity[0, 0] for p in planes],
            [p.velocity[1, 0] for p in planes], color='r')

        if not observed:
            plt.quiver(icebergs.positions[0, :], icebergs.positions[1, :],
                       icebergs.velocities[0, :], icebergs.velocities[1, :])
        else:
            plt.quiver(icebergs.positions[0, observed],
                       icebergs.positions[1, observed],
                       icebergs.velocities[0, observed],
                       icebergs.velocities[1, observed])

        for p in planes:
            if p.plot:
                plt.gca().add_patch(patches.Polygon(
                    p.fov, closed=True, fill=True, color='b', alpha=0.4))
                hist_x = [float(h[0]) for h in p.history]
                hist_y = [float(h[1]) for h in p.history]
                plt.plot(hist_x, hist_y, 'r')

                plt.quiver(icebergs.positions[0, p.findings],
                           icebergs.positions[1, p.findings],
                           icebergs.velocities[0, p.findings],
                           icebergs.velocities[1, p.findings],
                           color='g')

        tracked_targets = tracker.query_targets(rlim=plot_rlim)
        lmb.plot.plot_traces(tracked_targets, covellipse=False, max_back=1, alpha=0.8)

        plt.xlim([xmin, xmax])
        plt.ylim([xmin, xmax])

    plt.figure(figsize=(4 * PLOTS_WIDE, 4 * simlen // (PLOTS_WIDE * PLOT_EVERY)))
    tracker = init_tracker()
    icebergs = init_ice()
    planes = init_planes()
    found_icebergs = set()

    for k in range(simlen):
        print(k)
        plt.subplot(simlen / 5, 5, k + 1)
        if k > 0:
            update_ice(icebergs)
            tracker.predict()

        update_planes(planes, k)
        look_for_ice(planes, icebergs, k)
        found_icebergs |= {i for p in planes for i in p.findings}
        print("Enof:", tracker.enof_targets())

        for p in planes:
            if p.active(k):
                tracker.register_scan(p.scan)

        if k % PLOT_EVERY == 0:
            plt.subplot(simlen // (PLOTS_WIDE * PLOT_EVERY), PLOTS_WIDE, (k // PLOT_EVERY) + 1)
            plt.gca().set_title('t={} s'.format(k))
            mkplot()

    plt.gcf().savefig('icesetup.png', bbox_inches='tight')

    plt.figure()
    mkplot()
    plt.gca().set_title('t={}'.format(k))
    plt.gcf().savefig('icesetupfinal.png', bbox_inches='tight')

    tracked_targets = tracker.query_targets()
    with open("tracker.data", 'wb') as file_:
        pickle.dump(tracked_targets, file_)

def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    draw()


if __name__ == '__main__':
    main(*sys.argv[1:])
