import ast
import pytest

from unit_static_analyser.checker.checker import UnitChecker
from unit_static_analyser.units import Unit

# Define unit instances at module scope
m_unit = Unit.from_string("m")
s_unit = Unit.from_string("s")
kg_unit = Unit.from_string("kg")

def run_checker(code: str) -> UnitChecker:
    tree = ast.parse(code)
    checker = UnitChecker()
    checker.visit(tree)
    return checker

def assert_error(error, code, msg_contains=None):
    assert error.code == code
    if msg_contains:
        assert msg_contains in error.message

def test_assignment():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"] = 1
""")
    assert checker.units["a"] == m_unit

def test_addition():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"] = 1
b: Annotated[int, "m"] = 2
c = a + b
""")
    for name in ("a", "b", "c"):
        assert checker.units[name] == m_unit
    assert not checker.errors

def test_addition_error():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"] = 1
b: Annotated[int, "s"] = 2
c = a + b
""")
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")

def test_division_ok():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"] = 1
b: Annotated[int, "s"] = 2
c = a / b
""")
    assert checker.units["c"] == Unit.from_string("m.s^-1")

def test_disallow_missing_units():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"] = 1
b = a + 4
""")
    assert_error(checker.errors[0], "U002", "Operands must both have units")

def test_function_scope():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
def f():
    a: Annotated[int, "s"]
""")
    assert checker.units["a"] == m_unit
    assert checker.units["f.a"] == s_unit

def test_nested_function_scope():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
def f():
    a: Annotated[int, "s"]
    def g():
        a: Annotated[int, "kg"]
""")
    assert checker.units["a"] == m_unit
    assert checker.units["f.a"] == s_unit
    assert checker.units["f.g.a"] == kg_unit

def test_nested_scope_variable_lookup():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
def f():
    b: Annotated[int, "s"]
    def g():
        c = a + b
""")
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")

def test_nested_scope_variable_lookup_override():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
def f():
    b: Annotated[int, "m"]
    def g():
        b: Annotated[int, "s"]
        c = a + b
""")
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")
