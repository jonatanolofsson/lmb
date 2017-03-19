"""Library implementing the Labelled Multi-Bernoulli Filter."""

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
import sqlite3
import pickle
from collections import defaultdict
import itertools
import numpy as np
import numbers
from scipy.linalg import block_diag
from math import isnan
import threading
import queue

from .utils import connected_components
from .lmb_filter import predict, correct, threader
from .target import Target
from .pf import PF
from . import models


class DefaultTargetInit:
    """Default target initiator."""

    def __init__(self, q, pv, pS, dT=1):
        """Init."""
        self.q = q
        self.pv = np.eye(2) * pv if isinstance(pv, numbers.Number) else pv
        self.pS = pS
        self.dT = dT

    def __call__(self, tracker, report):
        """Init new target from report."""
        r = min(report.rB * tracker.params.lambdaB(report),
                tracker.params.rBmax, 1.0)
        print("New target:", r, "lambdaB", tracker.params.lambdaB(report))
        if r < 1e-6:
            return None
        id_ = tracker.new_id()
        model = models.ConstantVelocityModel(self.q, self.pS)

        x0 = np.array([report.z[0], report.z[1], 0.0, 0.0])
        P0 = block_diag(report.R, self.pv)
        pdf = PF.from_gaussian(x0, P0, tracker.params.N_max)

        return Target(id_, model, r, pdf)


class Parameters:
    """Cluster parmeters."""

    def __init__(self, **kwargs):
        """Init."""
        for name, value in kwargs.items():
            self.__dict__[name] = value

Parameters.N_max = 10000
Parameters.rBmax = 0.8
Parameters.lambdaB = staticmethod(lambda r: r.sensor.lambdaB)
Parameters.kappa = models.UniformClutter(0.01)
Parameters.init_target = DefaultTargetInit(1, 1, 1)
Parameters.w_lim = 1e-4
Parameters.maxhyp = 1e3
Parameters.r_lim = 1e-3
Parameters.nstd = 2
Parameters.N_PRED_THREADS = os.cpu_count()
Parameters.N_CORR_THREADS = os.cpu_count()


class LMB:
    """LMB class."""

    def __init__(self, params=None, dbfile=':memory:'):
        """Init."""
        self.params = params if params else Parameters()

        self.dbfile = dbfile
        self.dbc = sqlite3.connect(dbfile)
        self.db = self.dbc.cursor()
        self._init_db()

        self.HALT = threading.Event()

        self.predict_queue = queue.Queue()
        self.predict_threads = [
            threading.Thread(target=threader, args=(self, self.predict_queue, predict))
            for _ in range(self.params.N_PRED_THREADS)]
        for t in self.predict_threads:
            t.start()

        self.correct_queue = queue.Queue()
        self.correct_threads = [
            threading.Thread(target=threader, args=(self, self.correct_queue, correct))
            for _ in range(self.params.N_CORR_THREADS)]
        for t in self.correct_threads:
            t.start()

    def __enter__(self):
        """Enter."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit and clean up."""
        self.HALT.set()
        for t in self.predict_threads:
            t.join()
        for t in self.correct_threads:
            t.join()

    def _init_db(self):
        """Init database."""
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS targets ("
            "id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,"
            "r      REAL,"
            "data   BLOB"
            ");")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS target_index"
                        " USING rtree(id, min_x, max_x, min_y, max_y);")

    def new_id(self):
        """Register a new target id."""
        return self.db.execute("INSERT INTO targets DEFAULT VALUES;").lastrowid

    def query_targets(self, aa_bbox=None, rlim=None):
        """Get targets intersecting boundingbox."""
        if aa_bbox is None:
            rlimtxt = "" if rlim is None else ' WHERE r >= {}'.format(rlim)
            pickles = self.db.execute("SELECT data FROM targets" + rlimtxt)
        else:
            rlimtxt = "" if rlim is None else ' AND targets.r >= {}'.format(rlim)
            def get_targets(aa_bbox):
                #  PySQLite standard formatting doesn't work for some
                #  reason.. bug? Using .format instead, since known data.
                return self.db.execute((
                    "SELECT targets.data FROM targets "
                    "INNER JOIN target_index "
                    "ON targets.id = target_index.id WHERE "
                    "target_index.max_x >= {} AND "
                    "target_index.max_y >= {} AND "
                    "target_index.min_x <= {} AND "
                    "target_index.min_y <= {}" + rlimtxt +
                    ";").format(*aa_bbox))

            # FIXME: Use multiple queries if around wrapping-points!
            pickles = get_targets(aa_bbox)
        return {pickle.loads(p[0]) for p in pickles}

    def nof_targets(self, rlim=0, aa_bbox=None):
        """Count expected number of targets in region."""
        if aa_bbox is None:
            c = self.db.execute("SELECT COUNT(*) FROM targets WHERE r > ?;",
                                (rlim,))
        else:
            c = self.db.execute(("SELECT COUNT(*) FROM targets "
                                 "INNER JOIN target_index "
                                 "ON targets.id = target_index.id WHERE "
                                 "targets.r > ? AND "
                                 "target_index.max_x >= {} AND "
                                 "target_index.max_y >= {} AND "
                                 "target_index.min_x <= {} AND "
                                 "target_index.min_y <= {}"
                                 ";").format(*aa_bbox), (rlim,))
        res = c.fetchone()[0]
        return 0 if res is None else res

    def enof_targets(self, aa_bbox=None):
        """Count expected number of targets in region."""
        if aa_bbox is None:
            c = self.db.execute("SELECT SUM(r) FROM targets;")
        else:
            c = self.db.execute(("SELECT SUM(targets.r) FROM targets "
                                 "INNER JOIN target_index "
                                 "ON targets.id = target_index.id WHERE "
                                 "target_index.max_x >= {} AND "
                                 "target_index.max_y >= {} AND "
                                 "target_index.min_x <= {} AND "
                                 "target_index.min_y <= {}"
                                 ";").format(*aa_bbox))
        res = c.fetchone()[0]
        return 0 if res is None else res

    def _overlapping_targets(self, targets, r, nstd=2):
        """Select targets within reasonable range."""
        rbox = r.bbox(nstd)
        return {t for t in targets if t.bbox(nstd).intersects(rbox)}

    def _cluster(self, scan):
        """Collect clusters of targets."""
        active_targets = self.query_targets(scan.sensor.aa_bbox())

        targets = defaultdict(set)
        reports = defaultdict(set)
        connections = {r: set() for r in scan.reports}
        for r in scan.reports:
            targets[r] = self._overlapping_targets(active_targets, r,
                                                   self.params.nstd)
            for t in targets[r]:
                reports[t].add(r)
        for r in scan.reports:
            for t in targets[r]:
                connections[r].update(reports[t])

        for cluster_reports in connected_components(connections):
            yield (list(set().union(*(targets[r] for r in cluster_reports))),
                   list(cluster_reports))

        connected_targets = set().union(*(ts for ts in targets.values()))
        for t in active_targets - connected_targets:
            yield ([t], [])

    def birth(self, reports):
        """Add newborn targets to filter."""
        bsum = sum((1 - r.ruk) for r in reports)
        if bsum > 1e-9:
            for r in reports:
                r.rB = (1 - r.ruk) / bsum
        else:
            return
        self._save_targets(
            (self.params.init_target(self, r) for r in reports))

    def _save_targets(self, targets):
        """Store cluster data in database."""
        for t in targets:
            if t is not None:
                if any(isnan(p) for p in t.aa_bbox()):
                    print("NAN!:", target, target.pdf)
                    exit()
                self.db.execute(("REPLACE INTO target_index "
                                 "(id, min_x, min_y, max_x, max_y) "
                                 "VALUES ({}, {}, {}, {}, {});"
                                 ).format(t.id, *t.aa_bbox()))
                self.db.execute("UPDATE targets SET r=?, data=? "
                                "WHERE id=?", (t.r, pickle.dumps(t), t.id))
        self.dbc.commit()

    def _kill_targets(self, targets):
        """Kill off targets with low probability."""
        remove = {t for t in targets if t.r < self.params.r_lim}
        rids = ', '.join(str(t.id) for t in remove)
        self.db.execute("DELETE FROM targets WHERE id IN ({});".format(rids))
        self.db.execute("DELETE FROM target_index WHERE id IN ({});"
                        .format(rids))
        self.dbc.commit()
        targets -= remove

    def predict(self, dT=1, aa_bbox=None):
        """Move to next timestep."""
        targets = self.query_targets(aa_bbox)
        for t in targets:
            self.predict_queue.put((t, dT))

        self.predict_queue.join()

        self._kill_targets(targets)
        self._save_targets(targets)

    def register_scan(self, scan):
        """Register new scan."""
        targets = []
        stats = queue.Queue()
        for ctargets, creports in self._cluster(scan):
            self.correct_queue.put((ctargets, creports, scan.sensor, stats))
            targets += ctargets

        self.correct_queue.join()

        targets = set(targets)
        self._kill_targets(targets)
        self._save_targets(targets)
        self.birth(scan.reports)
        return list(stats.queue)
