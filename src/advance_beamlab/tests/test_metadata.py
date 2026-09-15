"""Tests for packaging metadata that cannot derive itself."""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang <muzhi.wang@ucl.ac.uk>
#          Jade Serfaty <jade.serfaty.17@ucl.ac.uk>
# License: BSD-3-Clause

import re
from pathlib import Path

import pytest

import advance_beamlab

ROOT = Path(__file__).resolve().parents[3]


def _repo_file(name):
    """Return a repository file, or skip when running from an installed wheel."""
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not present; running outside a source checkout")
    return path


def test_the_citation_file_states_the_package_version():
    """CITATION.cff is static YAML, so nothing makes it follow the package.

    ``pyproject.toml`` reads the version from ``__init__`` and ``doc/conf.py``
    reads it from the imported package, so those three cannot disagree. The
    citation file can, and a citation that names the wrong version is worse than
    one that names none.
    """
    text = _repo_file("CITATION.cff").read_text()
    match = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', text, re.M)
    assert match, "CITATION.cff has no version field"
    assert match.group(1).strip() == advance_beamlab.__version__


def test_the_version_is_declared_in_exactly_one_place():
    """Only ``__init__`` may state the version literally.

    ``pyproject.toml`` declares it dynamic and points at that file. If a literal
    version reappears in pyproject, the two can drift apart silently and the
    built distribution stops matching the package it contains.
    """
    text = _repo_file("pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text
    assert re.search(r'^version\s*=\s*"', text, re.M) is None, (
        "pyproject.toml declares a literal version; it should stay dynamic"
    )
