"""Study-scoped portable lock mode; leave captured shared engines unchanged."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def lock(path, *, shared=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process holds this run/member lock: {path}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def install():
    import s3t.artifacts as artifacts
    artifacts.lock = lock
    import s3t.engine as engine
    engine.lock = lock
