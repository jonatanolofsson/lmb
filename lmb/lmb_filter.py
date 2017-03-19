"""Library implementing Multiple Hypothesis Tracking."""

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
from math import exp
import queue

from .utils import LARGE
from .hypgen import murty


def threader(self, q, callback):
    """Threader helper."""
    while not self.HALT.is_set():
        try:
            args = q.get(timeout=0.1)
            callback(self, *args)
            q.task_done()
        except queue.Empty:
            pass


def predict(self, target, dT):
    """Perform parallel time update on cluster."""
    target.predict(self.params, dT)


def correct(self, targets, reports, sensor, stats):
    """Update cluster from multithread process."""
    N = len(targets)
    M = len(reports)
    # print("Cluster:", N, M)
    # print("Targets:", targets)

    if N == 0:
        stats.put({"nhyps": 0})
        return

    C = np.full((N, M + 2 * N), LARGE)
    for i, t in enumerate(targets):
        C[i, range(M)] = t.match(self.params, sensor, reports)
        C[i, M + i] = t.missed()
        C[i, M + N + i] = t.false()

    weights = np.zeros((N, M + 1))
    w_sum = 0

    nhyps = 0
    for score, assignment in murty(C):
        nhyps += 1
        assignment = np.array(assignment)
        w = exp(-score)
        w_sum += w
        ind = assignment < M + N
        assignment[assignment >= M] = M
        weights[ind, assignment[ind]] += w

        if w / w_sum < self.params.w_lim or nhyps >= self.params.maxhyp:
            break
    print("nhyps:", nhyps, "tracks:", N, "meas:", M)

    weights /= w_sum

    for i, t in enumerate(targets):
        t.correct(weights[i, ])

    for r, ruk in zip(reports, np.sum(weights, axis=0)):
        r.ruk = ruk

    stats.put({"nhyps": nhyps})
