import ast

import pytest

from unit_static_analyser.checker.checker import UnitChecker
from unit_static_analyser.units.types import m, s


def test_assignment():
    """Test that a variable can be assigned a unit type."""
    code = """
from typing import Annotated
from units.types import m
a: Annotated[int, m] = 1
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    checker.visit(tree)
    assert checker.units["a"] is m


def test_addition_error():
    """Test that adding two quantities with different units raises a TypeError."""
    code = """
from typing import Annotated
from units.types import m, s
a: Annotated[int, m] = 1
b: Annotated[int, s] = 2
c = a + b
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    with pytest.raises(TypeError):
        checker.visit(tree)


def test_division_ok():
    """Test that dividing two quantities with compatible units works."""
    code = """
from typing import Annotated
from units.types import m, s
a: Annotated[int, m] = 1
b: Annotated[int, s] = 2
c = a / b
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    checker.visit(tree)
    assert "c" in checker.units
    assert checker.units["c"] is m / s


def test_disallow_missing_units():
    """Test that operations involving variables without units raise a TypeError."""
    code = """
from typing import Annotated
from units.types import m
a: Annotated[int, m] = 1
b = a + 4
"""
    tree = ast.parse(code)
    checker = UnitChecker()
    with pytest.raises(TypeError):
        checker.visit(tree)
