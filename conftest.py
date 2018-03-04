"""Configuration for automated test suite, allows you to automatically
skip tests marked as slow unless you provide the --slow option
on the commandline when invoking pytest
"""

import pytest

def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run slow tests")
    parser.addoption("--notravis", action="store_true",
        help="Run tests that travis doesn't run by default. These tests "\
             "typically require substantial data that would need to be "\
             "fetched from elsewhere.")
    parser.addoption("--experiment", action="store_true",
        help="Runs end-to-end experiments. This is for reproducing reported "\
             "results and to ensure models converge with larger amounts of "\
             "data.")
    parser.addoption("--preprocess", action="store_true",
        help="Encourages re-preprocessing of data.")

def pytest_runtest_setup(item):
    """
    Skip tests if they are marked as slow, unless the
    test run is explicitly specified to run the slow tests.

    A similar marker `notravis` is used for tests that aren't that slow but
    require data unavailable to Travis. This includes datasets that aren't open
    or would be a bit time consuming to download even though they are available
    locally.

    The 'experiment' marker is for experiments where error rate results need to
    demonstrate adequate convergence. They are typically long-running, and the
    whole suite would take days or weeks to run if all the experiments were
    done together.
    """

    if 'slow' in item.keywords and not item.config.getoption("--slow"):
        pytest.skip("need --slow option to run")
    if 'notravis' in item.keywords and not item.config.getoption("--notravis"):
        pytest.skip("need --notravis option to run")
    if 'experiment' in item.keywords and not item.config.getoption("--experiment"):
        pytest.skip("need --experiment option to run")
