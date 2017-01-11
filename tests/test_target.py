"""Test Target methods."""

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

import unittest
from unittest.mock import MagicMock
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lmb.target import Target


class TestTarget(unittest.TestCase):
    """Target method tests."""

    def setUp(self):
        """Set up."""
        self.filter = MagicMock()

    def test_initial(self):
        """Test creation of new target."""
        model = MagicMock()
        Target(0, model)
