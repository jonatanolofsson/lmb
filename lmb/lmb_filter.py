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

from .utils import LARGE, df
from .hypgen import murty


def predict(args):
    """Perform parallel time update on cluster."""
    (params, target, dT) = args
    target.predict(params, dT)
    return target


def correct(args):
    """Update cluster from multithread process."""
    (params, targets, reports, sensor) = args
    N = len(targets)
    M = len(reports)
    # print("Cluster:", N, M)
    # print("Targets:", targets)

    if N == 0:
        resdf = df()
        resdf.targets = targets
        resdf.reports = reports
        resdf.nhyps = 0
        return resdf

    C = np.full((N, M + 2 * N), LARGE)
    for i, t in enumerate(targets):
        C[i, range(M)] = t.match(params, sensor, reports)
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

        if w < params.w_lim:
            break
    print("nhyps:", nhyps)

    weights /= w_sum

    for i, t in enumerate(targets):
        t.correct(weights[i, ])

    for r, ruk in zip(reports, np.sum(weights, axis=0)):
        r.ruk = ruk

    resdf = df()
    resdf.targets = targets
    resdf.reports = reports
    resdf.nhyps = nhyps
    return resdf
