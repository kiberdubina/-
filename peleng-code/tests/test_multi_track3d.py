import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.profile import load_patterns
from app.session import Session
from app.source import SyntheticSource


class MultiTrack3DTest(unittest.TestCase):
    def _session(self) -> Session:
        profile = load_patterns(ROOT / "config.yaml")["green_ball"]
        return Session(profile, SyntheticSource())

    def test_far_apart_xyz_creates_two_tracks(self):
        session = self._session()
        session._update_tracks3d(np.array([1.0, 0.0, 1.0]), 0.0, {})
        session._update_tracks3d(np.array([5.0, 0.0, 1.0]), 0.04, {})
        self.assertEqual(len(session.tracks3d), 2)
        self.assertEqual(session.target_id, 2)
        self.assertTrue(session.tracks3d[0].ready)
        self.assertTrue(session.tracks3d[1].ready)
        self.assertAlmostEqual(float(session.tracks3d[1].kf.statePost[0, 0]), 5.0, places=3)

    def test_near_xyz_updates_first_track(self):
        session = self._session()
        session._update_tracks3d(np.array([1.0, 0.0, 1.0]), 0.0, {})
        session._update_tracks3d(np.array([1.05, 0.0, 1.0]), 0.04, {})
        self.assertEqual(len(session.tracks3d), 1)
        self.assertEqual(session.target_id, 1)
        self.assertAlmostEqual(float(session.tracks3d[0].kf.statePost[0, 0]), 1.05, places=2)


if __name__ == "__main__":
    unittest.main()
