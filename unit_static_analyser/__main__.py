"""Entrypoint that runs against any filenames provided on the command line."""

from pathlib import Path

from .mypy_checker.checker import UnitChecker


def run() -> None:
    """Run the unit checker on provided filenames."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m unit_static_analyser <file1.py> <file2.py> ...")
        sys.exit(1)

    checker = UnitChecker()
    checker.check(
        [path for path_str in sys.argv[1:] if (path := Path(path_str)).exists()]
    )

    print("Unit checking completed.")
    print(f"Errors: {checker.errors}")


if __name__ == "__main__":
    run()
