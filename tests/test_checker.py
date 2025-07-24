import ast

import pytest

from unit_static_analyser.checker.checker import UnitChecker
from unit_static_analyser.units import m, s


def test_assignment():
    """Test that a variable can be assigned a unit type."""
    code = """
from typing import Annotated
from units import m
a: Annotated[int, m] = 1
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    checker.visit(tree)
    assert checker.units["a"] == m


def test_addition_error():
    """Test that adding two quantities with different units raises a TypeError."""
    code = """
from typing import Annotated
from units import m, s
a: Annotated[int, m] = 1
b: Annotated[int, s] = 2
c = a + b
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    checker.visit(tree)
    error = checker.errors[0]
    error.code = "U001"
    error.lineno = 5
    error.message = f"Cannot add operands with different units: m and s"

def test_division_ok():
    """Test that dividing two quantities with compatible units works."""
    code = """
from typing import Annotated
from units import m, s
a: Annotated[int, m] = 1
b: Annotated[int, s] = 2
c = a / b
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    checker.visit(tree)
    assert "c" in checker.units
    assert checker.units["c"] == m / s


def test_disallow_missing_units():
    """Test that operations involving variables without units raise a TypeError."""
    code = """
from typing import Annotated
from unit_static_analyser.units import m
a: Annotated[int, m] = 1
b = a + 4
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    checker.visit(tree)
    error = checker.errors[0]
    error.code = "U002"
    error.lineno = 4
    error.message = f"Operands must both have units"
