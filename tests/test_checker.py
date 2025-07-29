import ast
import pytest

from unit_static_analyser.checker.checker import UnitChecker
from unit_static_analyser.units import Unit

# Define unit instances at module scope
m_unit = Unit.from_string("m")
s_unit = Unit.from_string("s")
kg_unit = Unit.from_string("kg")


def run_checker(code: str, module_name="__main__", external_units=None) -> UnitChecker:
    tree = ast.parse(code)
    checker = UnitChecker(module_name=module_name)
    if external_units:
        checker.units.update(external_units)
    checker.visit(tree)
    return checker


def assert_error(error, code, msg_contains=None):
    assert error.code == code
    if msg_contains:
        assert msg_contains in error.message


def assert_u005_error(error, returned_unit, expected_unit):
    assert error.code == "U005"
    assert error.message.startswith(
        "Units of returned value does not match function signature"
    )
    assert f"returned={returned_unit}" in error.message
    assert f"expected={expected_unit}" in error.message


def test_assignment():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"] = 1
""")
    assert checker.units["__main__.a"] == m_unit


def test_addition():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"] = 1
b: Annotated[int, "m"] = 2
c = a + b
""")
    for name in ("a", "b", "c"):
        assert checker.units[f"__main__.{name}"] == m_unit
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
    assert checker.units["__main__.c"] == Unit.from_string("m.s^-1")


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
    assert checker.units["__main__.a"] == m_unit
    assert checker.units["__main__.f.a"] == s_unit


def test_nested_function_scope():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
def f():
    a: Annotated[int, "s"]
    def g():
        a: Annotated[int, "kg"]
""")
    assert checker.units["__main__.a"] == m_unit
    assert checker.units["__main__.f.a"] == s_unit
    assert checker.units["__main__.f.g.a"] == kg_unit


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


def test_class_scope():
    """Test that class variables are scoped correctly and can be accessed."""
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
""")
    assert checker.units["__main__.A.a"] == m_unit


def test_class_variable_lookup():
    """Test that class variables can be accessed and used in expressions."""
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
b: Annotated[int, "m"]
c = A.a + b
""")
    assert checker.units["__main__.c"] == m_unit


def test_class_variable_lookup_nested():
    """Test that class variables can be accessed in nested classes."""
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
    class B(self):
        b: Annotated[int, "m"]
c = A.a + A.B.b
""")
    assert checker.units["__main__.c"] == m_unit


def test_class_inheritance():
    """Test that class inheritance works correctly with unit annotations."""
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
class B(A):
    b: Annotated[int, "s"]
""")
    assert checker.units["__main__.B.a"] == m_unit
    assert checker.units["__main__.B.b"] == s_unit


def test_instance_variable_lookup():
    """Test that instance variables can be accessed and used in expressions."""
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"] = 1
a = A()
b: Annotated[int, "m"]
c = a.a + b
""")
    assert checker.units["__main__.c"] == m_unit


def test_instance_variable_lookup_rename():
    """Test that instance variables can be accessed and used in expressions."""
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"] = 1
a = A()
b = a
c: Annotated[int, "m"]
d = b.a + c
""")
    assert checker.units["__main__.d"] == m_unit


def test_instance_variable_lookup_error():
    """Test that instance variables can be accessed and used in expressions."""
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"] = 1
a = A()
b = a
c: Annotated[int, "s"]
d = b.a + c
""")
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_function_return_value():
    """Test that function return values are handled correctly."""
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    a: Annotated[int, "m"]
    return a
b = f()
""")
    assert checker.units["__main__.b"] == m_unit


def test_function_return_value_mismatch():
    """Test that an error is reported if the wrong unit is returned from a function."""
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    a: Annotated[int, "s"]
    return a
""")
    assert_u005_error(
        checker.errors[0],
        returned_unit="s",
        expected_unit="m",
    )


def test_function_return_value_missing_units():
    """Test that function signatures with return values are handled correctly."""
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    return 1
""")
    assert_u005_error(
        checker.errors[0],
        returned_unit="None",
        expected_unit="m",
    )


def test_function_return_value_nested():
    """Test that function signatures with return values are handled correctly."""
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    def f() -> Annotated[int, "m"]:
        a: Annotated[int, "m"]
        return a
    return f()
b = f()
""")
    assert checker.units["__main__.b"] == m_unit


def test_function_return_value_nested_different_signatures():
    """Test that function signatures with return values are handled correctly."""
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    def f() -> Annotated[int, "s"]:
        a: Annotated[int, "s"]
        return a
    a: Annotated[int, "m"]
    return a
b = f()
""")
    assert checker.units["__main__.b"] == m_unit


def test_function_return_value_no_unit_in_signature():
    """Test that function signatures with return values are handled correctly."""
    checker = run_checker("""
from typing import Annotated
def f() -> int:
    a: Annotated[int, "m"]
    return a
""")
    assert_u005_error(
        checker.errors[0],
        returned_unit="m",
        expected_unit="None",
    )


def test_module_import():
    """Test that imported units are recognized."""
    other_checker = run_checker(
        """
from typing import Annotated
var_with_units: Annotated[int, "m"]
""",
        module_name="other_module",
    )
    checker = run_checker(
        """
from typing import Annotated
from other_module import var_with_units
""",
        external_units=other_checker.units,
    )
    assert checker.units["__main__.var_with_units"] == m_unit
    assert checker.units["other_module.var_with_units"] == m_unit
