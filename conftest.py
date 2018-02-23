"""Configuration for unit test suite, allows you to automatically
skip tests marked as slow unless you provide the --runslow option
on the commandline when invoking pytest
"""

import pytest

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--notravis", action="store_true",
        help="run tests that travis doesn't run by default")

def pytest_runtest_setup(item):
    """
    Skip tests if they are marked as slow, unless the
    test run is explicitly specified to run the slow tests.

    A similar marker `notravis` is used for tests that aren't that slow but
    require data unavailable to Travis. This includes datasets that aren't open
    or would be a bit time consuming to download even though they are available
    locally.
    """
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
    if 'notravis' in item.keywords and not item.config.getoption("--notravis"):
        pytest.skip("need --notravis option to run")
