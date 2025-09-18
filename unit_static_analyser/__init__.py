"""The main module for Unit Static Analyser."""

import argparse
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .checker.checker import UnitChecker

with suppress(PackageNotFoundError):
    __version__ = version(__name__)


def run() -> None:
    """Run the unit checker on provided filenames with optional unit output."""
    parser = argparse.ArgumentParser(
        description="Unit Static Analyser: Check units in Python files."
    )
    parser.add_argument(
        "files",
        metavar="file_or_directory",
        nargs="+",
        help="Python files or directories to check",
    )
    parser.add_argument(
        "-u",
        "--show-units",
        action="store_true",
        help="Show unit data for all checked variables",
    )
    args = parser.parse_args()

    checker = UnitChecker()
    checker.check(
        [path for path_str in args.files if (path := Path(path_str)).exists()]
    )

    print("Unit checking completed.")
    print(f"Errors: {checker.errors}")
    if args.show_units:
        print("Units:")
        for key, val in checker.units.items():
            print(f"{key.fullname}: {val}")
