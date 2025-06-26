"""Tests for the main module."""

from unit_static_analyser import __version__


def test_version():
    """Check that the version is acceptable."""
    assert isinstance(__version__, str)
