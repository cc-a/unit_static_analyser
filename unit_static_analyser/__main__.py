"""Entrypoint that runs against any filenames provided on the command line."""

from .checker.checker import UnitChecker


def run() -> None:
    """Run the unit checker on provided filenames."""
    import ast
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m unit_static_analyser <file1.py> <file2.py> ...")
        sys.exit(1)

    checker = UnitChecker()

    for filename in sys.argv[1:]:
        with open(filename) as file:
            code = file.read()
            tree = ast.parse(code)
            checker.visit(tree)

    print("Unit checking completed.")
    print("Units found:", checker.units)


if __name__ == "__main__":
    run()
