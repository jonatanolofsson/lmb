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
# import cProfile

sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
import lmb


np.random.seed(1)


def draw():
    """Create plot."""
    plt.figure(figsize=(10, 10))
    params = lmb.Parameters()
    params.N_max = 50000
    params.kappa = lmb.models.UniformClutter(0.0001)
    params.init_target = lmb.DefaultTargetInit(0.1, 1, 1)
    params.r_lim = 0.02
    params.nstd = 1.9
    with lmb.LMB(params) as tracker:
        sensor = lmb.sensors.EyeOfMordor()
        sensor.lambdaB = 0.1
        targets = [
            np.array([0.0, 0.0,  1, 0.5]),
            np.array([0.0, 10.0, 1, -0.5]),
        ]
        ntargets_true = []
        ntargets_verified = []
        ntargets = []
        lmb.plot.plt.subplot(2, 1, 1)
        for k in range(30):
            print()
            print("k:", k)
            if k > 0:
                tracker.predict(1)
                for t in targets:
                    t[0:2] += t[2:]
            if k == 5:
                targets.append(np.array([5.0, 5.0, 1.0, 0.0]))
            if k % 7 == 0:
                targets.append(np.random.multivariate_normal(
                    np.array([k, 7.0, 0.0, 0.0]),
                    np.diag([0.5] * 4)))
            if k % 7 == 1:
                del targets[-1]
            if k == 10:
                targets.append(np.array([10.0, -30.0, 1.0, -0.5]))
            if k == 20:
                targets.append(np.array([k, 0.0, 1.0, 4.0]))

            reports = {lmb.GaussianReport(
                # np.random.multivariate_normal(t[0:2], np.diag([0.01] * 2)),  # noqa
                t[0:2],
                np.eye(2) * 0.5,
                lmb.models.position_measurement,
                i)
                for i, t in enumerate(targets)}
            this_scan = lmb.Scan(sensor, reports)
            tracker.register_scan(this_scan)
            ntargets.append(tracker.enof_targets())
            ntargets_verified.append(tracker.nof_targets(0.7))
            ntargets_true.append(len(targets))
            lmb.plot.plot_scan(this_scan)
            plt.plot([t[0] for t in targets],
                     [t[1] for t in targets],
                     marker='D', color='y', alpha=.5, linestyle='None')
        lmb.plot.plot_traces(tracker.query_targets(), covellipse=True)
        lmb.plot.plt.axis([-1, k + 1, -k - 1, k + 1 + 10])
        lmb.plot.plt.ylabel('Tracks')
        lmb.plot.plt.subplot(2, 1, 2)
        lmb.plot.plt.plot(ntargets, label='Estimate')
        lmb.plot.plt.plot(ntargets_true, label='True')
        lmb.plot.plt.plot(ntargets_verified, label='Verified')
        lmb.plot.plt.ylabel('# Targets')
        plt.legend(fancybox=True, framealpha=0.5, loc=4, prop={'size': 10})
        lmb.plot.plt.axis([-1, k + 1, min(ntargets + ntargets_true) - 0.1,
                           max(ntargets + ntargets_true) + 0.1])


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
