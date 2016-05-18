"""Methods to handle MHT target."""

from copy import deepcopy


class Target:
    """Class to represent a single MHT target."""

    def __init__(self, filter, score):
        """Init."""
        self.tracks = [Track(
            None, None, self, filter=filter, initial_score=score)]
        self.reset()

    def assign(self, parent, m):
        """Assign measurement to track node to expand tree."""
        print('Assigning', m, 'to', parent)
        if m not in self.new_tracks:
            self.new_tracks[m] = Track(m, parent, self)
        return self.new_tracks[m]

    def predict(self, dT):
        """Move to next time step."""
        self.tracks = self.new_tracks.values()
        for track in self.tracks:
            track.predict(dT)
        self.reset()

    def reset(self):
        """Reset caches etc."""
        self.new_tracks = {}
        self._score_cache = {}

    def score(self, report):
        """Return the score for a given report, for all tracks."""
        if report not in self._score_cache:
            self._score_cache[report] = [(track.score(report), track)
                                         for track in self.tracks]
        return self._score_cache[report]

    def __repr__(self):
        """String representation of object."""
        return "T(0x{:x})".format(id(self))


class Track:
    """Class to represent the tracks in a target tree."""

    def __init__(self, m, parent_track, target,
                 filter=None, initial_score=None):
        """Init."""
        self.parent_track_id = id(parent_track) if parent_track else id(target)
        self.filter = filter or deepcopy(parent_track.filter)
        if m:
            self.my_score = self.filter.correct(m)
        else:
            self.my_score = initial_score
        self.target = target

    def is_new_target(self):
        """Check if target is brand new."""
        return id(self.target) == self.parent_track_id

    def assign(self, m):
        """Assign measurement to track."""
        if self.is_new_target():
            return self
        return self.target.assign(self, m)

    def predict(self, dT):
        """Move to next time step."""
        self.filter.predict(dT)

    def score(self, m=None):
        """Find the score of assigning a measurement to the filter."""
        if m is None:
            return self.my_score
        return self.filter.score(m)

    def __repr__(self):
        """Return string representation of object."""
        return "Tr(0x{:x}: {})".format(
            id(self),
            "[{}]".format(", ".join(str(int(x)) for x in self.filter.x)))

    def __lt__(self, b):
        """Check if self < b."""
        return id(self) < id(b)