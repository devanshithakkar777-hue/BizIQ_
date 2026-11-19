"""One place that sets up logging for the app and scripts. Level via BIZIQ_LOG_LEVEL (default INFO); output goes to stderr (Docker/systemd collect it)."""
import logging
import os

_done = False


def setup() -> logging.Logger:
    global _done
    log = logging.getLogger("biziq")
    if not _done:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(h)
        log.setLevel(os.environ.get("BIZIQ_LOG_LEVEL", "INFO").upper())
        log.propagate = False
        _done = True
    return log
