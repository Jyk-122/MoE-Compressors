"""Console logs carry context while command results remain on stdout."""

import json
import os
import re
import subprocess
import sys

import pytest


@pytest.mark.parametrize("level", [None, "warning"])
def test_logging_format_level_and_stdout(level):
    env = dict(os.environ, RANK="3", PYTHONIOENCODING="utf-8")
    env.pop("PREFETCH_LOG_LEVEL", None)
    if level:
        env["PREFETCH_LOG_LEVEL"] = level
    result = subprocess.run([sys.executable, "-c", """
import logging
from prefetch.utils.logging import configure_logging
configure_logging()
configure_logging()
logger = logging.getLogger('prefetch.test')
logger.info('Model loaded')
logger.warning('Skipping %s', 'missing.jpg')
print('{"result": 1}')
"""], capture_output=True, text=True, encoding="utf-8", env=env, check=True)
    assert json.loads(result.stdout) == {"result": 1}
    assert ("INFO [rank=3] prefetch.test: Model loaded" in result.stderr) == (level is None)
    assert result.stderr.count("WARNING [rank=3] prefetch.test: Skipping missing.jpg") == 1
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", result.stderr)


def test_imports_leave_host_logging_unchanged():
    subprocess.run([sys.executable, "-c", """
import logging
root = logging.getLogger()
handlers, level = list(root.handlers), root.level
import prefetch.utils.logging
import prefetch.datasets.prepare
import prefetch.evaluation.cache.simulate
assert root.handlers == handlers and root.level == level
"""], check=True, capture_output=True, text=True)
