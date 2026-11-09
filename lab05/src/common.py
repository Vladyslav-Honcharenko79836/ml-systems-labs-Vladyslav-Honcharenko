import contextlib
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee_results(filename):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    old = sys.stdout
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        sys.stdout = _Tee(old, f)
        try:
            yield path
        finally:
            sys.stdout = old
    print(f"[saved to {path.relative_to(LAB_DIR.parent).as_posix()}]")
