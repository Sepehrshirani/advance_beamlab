# Authors: Sepehr Shirani and Muzhi Wang <sepehrshirani@gmail.com>
# License: BSD-3-Clause
"""Shared pytest configuration, following MNE-Python's test conventions."""

import mne
import pytest

# Keep MNE quiet across the suite; individual tests raise the level when they
# assert on log output.
mne.set_log_level("ERROR")


def pytest_configure(config):
    """Register markers and turn stray warnings into failures."""
    config.addinivalue_line(
        "markers", "slowtest: a test that takes more than a few seconds"
    )
    # MNE-Python's CI runs with warnings as errors so that a new warning cannot
    # be introduced unnoticed. The allowlist below covers the warnings that come
    # from outside this package and that we cannot act on; ``-W error`` can then
    # be added on the command line (and is, in CI) to enforce the policy.
    for line in (
        # scikit-learn's shrunk-covariance cross-validation calls slogdet on
        # rank-deficient matrices; raised from sklearn, not from this package.
        "ignore:divide by zero encountered in slogdet:RuntimeWarning",
        "ignore:overflow encountered in slogdet:RuntimeWarning",
        "ignore:invalid value encountered in slogdet:RuntimeWarning",
        # Emitted by mne.compute_covariance for baseline-free simulated epochs.
        "ignore:Epochs are not baseline corrected:RuntimeWarning",
        # numpy/scipy deprecations from dependencies.
        "ignore:.*is deprecated.*:DeprecationWarning",
    ):
        config.addinivalue_line("filterwarnings", line)


@pytest.fixture(scope="session")
def sample_data_path():
    """Path to the MNE ``sample`` dataset, or skip if it is not downloaded."""
    try:
        path = mne.datasets.sample.data_path(download=False)
    except Exception:  # pragma: no cover - depends on local dataset state
        path = None
    if path is None or not (path / "MEG" / "sample").is_dir():
        pytest.skip("MNE sample dataset not downloaded")
    return path
