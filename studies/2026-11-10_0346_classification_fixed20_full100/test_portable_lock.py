"""Reproduce NFS shared read-lock descriptor requirements without a cluster."""
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studies.classification_fixed20_full100 import portable_lock


class LockTests(unittest.TestCase):
    def test_shared_lock_uses_readable_descriptor_and_does_not_change_engine(self):
        import s3t.artifacts as artifacts
        import s3t.engine as engine
        before = Path(artifacts.__file__).read_bytes()
        with tempfile.TemporaryDirectory() as tmp, patch.object(portable_lock.fcntl, 'flock', side_effect=fcntl.lockf):
            with portable_lock.lock(Path(tmp) / 'native/.lock', shared=True):
                pass
        with patch.object(artifacts, 'lock'), patch.object(engine, 'lock'):
            portable_lock.install()
            self.assertIs(engine.lock, portable_lock.lock)
        self.assertEqual(Path(artifacts.__file__).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
