from pathlib import Path

import pytest

from unit_static_analyser.checker.checker import UnitChecker
from unit_static_analyser.checker.errors import UnitCheckerError
from unit_static_analyser.units import Unit

# Define unit instances at module scope
m_unit = Unit.from_string("m")
s_unit = Unit.from_string("s")
kg_unit = Unit.from_string("kg")

TEST_MODULE_NAME = "test_module"


def check_unit(
    checker: UnitChecker,
    obj_path: str,
    unit: Unit,
    prefix: str = f"{TEST_MODULE_NAME}.",
):
    """Helper function for checking a unit in the test module."""
    for var, var_unit in checker.units.items():
        if var.fullname == f"{prefix}{obj_path}":
            assert unit == var_unit
            break
    else:
        raise ValueError(f"Missing unit - {obj_path}")


def run_checker(code: str, tmp_path: Path) -> UnitChecker:
    """Run the UnitChecker on the given code using a temp file via pytest's tmp_path.

    Args:
        code: The Python code to analyze.
        tmp_path: pytest's temporary directory fixture.
        module_name: The module name to use for the analysis.
        external_units: Optional dictionary of external units to include.

    Returns:
        The UnitChecker instance after analysis.
    """
    file_path = tmp_path / f"{TEST_MODULE_NAME}.py"
    file_path.write_text(code)
    checker = UnitChecker()
    checker.check([file_path])
    return checker


def assert_error(error: UnitCheckerError, code: str, lineno: int, message: str):
    """Assert that an error matches the expected code, message content, and line number.

    Args:
        error: The UnitCheckerError instance to check.
        code: The expected error code.
        lineno: Optional expected line number for the error.
        message: Optional substring that should be in the error message.
    """
    assert error.code == code
    assert error.lineno == lineno
    assert message == error.message


def assert_error_u001(
    error: UnitCheckerError, lineno: int, left_unit: Unit, right_unit: Unit
):
    """Assert U001 error for incompatible addition units."""
    expected_msg = (
        f"Cannot add operands with different units: {left_unit} and {right_unit}"
    )
    assert_error(error, "U001", lineno, expected_msg)


def assert_error_u002(error: UnitCheckerError, lineno: int):
    """Assert U002 error for missing units in operands."""
    expected_msg = "Operands must both have units"
    assert_error(error, "U002", lineno, expected_msg)


def assert_error_u003(
    error: UnitCheckerError,
    lineno: int,
    func_name: str,
    arg_num: int,
    received_unit: Unit,
    expected_unit: Unit,
):
    """Assert U003 error for function argument unit mismatch."""
    expected_msg = (
        f"Argument {arg_num} to function '{func_name}' has unit {received_unit}, "
        f"expected {expected_unit}"
    )
    assert_error(error, "U003", lineno, expected_msg)


def assert_error_u004(
    error: UnitCheckerError, lineno: int, returned: Unit, expected: Unit
):
    """Assert U004 error for return value unit mismatch."""
    expected_msg = (
        "Unit of return value does not match function signature: "
        f"returned {returned}, expected {expected}"
    )
    assert_error(error, "U004", lineno, expected_msg)


def assert_error_u005(
    error: UnitCheckerError, left_unit: Unit, right_unit: Unit, lineno: int
):
    """Assert that a U005 error matches expected left/right units and message."""
    expected_msg = (
        f"Cannot compare operands with different units: {left_unit} and {right_unit}"
    )
    assert_error(error, "U005", lineno, expected_msg)


def assert_error_u006(error: UnitCheckerError, lineno: int):
    """Assert that a U006 error matches expected left/right units and message."""
    expected_msg = "Cannot compare a unitful operand with a unitless operand"
    assert_error(error, "U006", lineno, expected_msg)


def assert_error_u007(
    error: UnitCheckerError, if_unit: Unit, else_unit: Unit, lineno: int
):
    """Assert that a U007 error matches expected if/else units and message."""
    expected_msg = (
        f"Conditional branches have different units: {if_unit} and {else_unit}"
    )
    assert_error(error, "U007", lineno, expected_msg)


def assert_error_u008(error: UnitCheckerError, lineno: int):
    """Assert that a U008 error matches expected message."""
    expected_msg = "Both branches of conditional must a unit."
    assert_error(error, "U008", lineno, expected_msg)


def assert_error_u009(error: UnitCheckerError, lineno: int):
    """Assert U009 error for non-integer exponent."""
    expected_msg = "Exponent must be an explicit integer value."
    assert_error(error, "U009", lineno, expected_msg)


def assert_error_u010(
    error: UnitCheckerError,
    lineno: int,
    var_name: str,
    expected_unit: Unit,
    received_unit: Unit | None,
):
    """Assert U010 error for incompatible assignment units."""
    expected_msg = (
        f"Incompatible unit in assignment to {var_name}: expected {expected_unit}, "
        f"received {received_unit}"
    )
    assert_error(error, "U010", lineno, expected_msg)


def assert_error_u011(error: UnitCheckerError, lineno: int):
    """Assert U011 error for variable unit override."""
    expected_msg = "Variable already has a unit"
    assert_error(error, "U011", lineno, expected_msg)


def assert_error_u012(error: UnitCheckerError, lineno: int, operator: str):
    """Assert U012 error for assignment operators."""
    expected_msg = f"Cannot use {operator}= operator on expressions with units."
    assert_error(error, "U012", lineno, expected_msg)


def test_assignment_arbitrary_expression(tmp_path: Path):
    """Test that units are correctly inferred for arbitrary complex expressions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
def f() -> A:
    return A()
b = f().a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_assignment_arbitrary_expression2(tmp_path: Path):
    """Another test that units are inferred for arbitrary complex expressions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
def f() -> A:
    return A
b = f().a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_assignment_arbitrary_expression3(tmp_path: Path):
    """Another test that units are inferred for arbitrary complex expressions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
def f() -> A:
    return A
def f2(a: Annotated[int, "unit:s"]) -> Annotated[int, "unit:s"]:
    return a
b = f2(f().a)
""",
        tmp_path,
    )
    check_unit(checker, "b", s_unit)
    assert_error_u003(checker.errors[0], 9, "test_module.f2", 1, m_unit, s_unit)


def test_function_return_instance(tmp_path: Path):
    """Test unit lookups when operating on a object returned by a function."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
    def f(self, b: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
        return self.a
def f() -> A:
    return A()
b: Annotated[int, "unit:s"] = 1
c = f().f(b)
d = f().a
""",
        tmp_path,
    )
    assert_error_u003(checker.errors[0], 10, "test_module.A.f", 1, s_unit, m_unit)
    check_unit(checker, "c", m_unit)
    check_unit(checker, "d", m_unit)


def test_assignment_instance_attribute(tmp_path: Path):
    """Test that units are correctly inferred for instance attribute access."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
a = A()
b = a.a
c = A().a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)
    check_unit(checker, "c", m_unit)


def test_assignment_member_access(tmp_path: Path):
    """Test that units are correctly inferred for class attribute access."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
b = A.a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_assignment_function_call(tmp_path: Path):
    """Test units are inferred for function return values with unit annotations."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> Annotated[int, "unit:m"]:
    pass
b = f()
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)
    node = next(node for node in checker.units if node.name == "f")
    assert checker.node_module_names[node] == TEST_MODULE_NAME


def test_assignment_class_init_member_access(tmp_path: Path):
    """Test units are inferred for attribute access on instances created inline."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
b = A().a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_assignment_alias(tmp_path: Path):
    """Test that units are correctly propagated through variable aliasing."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"]
b = a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_assignment(tmp_path: Path):
    """Test that units are correctly assigned to annotated variables."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
""",
        tmp_path,
    )
    check_unit(checker, "a", m_unit)
    ((node, module_name),) = checker.node_module_names.items()
    assert node.name == "a"
    assert module_name == TEST_MODULE_NAME


def test_addition(tmp_path: Path):
    """Test addition of variables with same unit is allowed and unit is preserved."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:m"] = 2
c = a + b
""",
        tmp_path,
    )
    for name in ("a", "b", "c"):
        check_unit(checker, name, m_unit)
    assert not checker.errors


def test_addition_error(tmp_path: Path):
    """Test that addition of variables with different units raises an error."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:s"] = 2
c = a + b
""",
        tmp_path,
    )
    assert_error_u001(checker.errors[0], 5, m_unit, s_unit)


def test_division_ok(tmp_path: Path):
    """Test division of variables with units produces the correct resulting unit."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:s"] = 2
c = a / b
""",
        tmp_path,
    )
    check_unit(checker, "c", Unit.from_string("m.s^-1"))


def test_disallow_missing_units(tmp_path: Path):
    """Test that operations involving unannotated values raise an error."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b = a + 4
""",
        tmp_path,
    )
    assert_error_u002(checker.errors[0], 4)


@pytest.mark.parametrize("symbol", ("+", "-"))
def test_expression_unary(symbol: str, tmp_path: Path):
    """Test that unary plus and minus preserve the unit of the operand."""
    checker = run_checker(
        f"""
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b = {symbol}a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_expression_instance_method(tmp_path: Path):
    """Test that units are correctly inferred for return values of instance methods."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"] = 1
    def f(self) -> Annotated[int, "unit:m"]:
        return self.a
b = A().f()
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_expression_nested_function_call_mismatch(tmp_path: Path):
    """Test that unit mismatches are detected in nested function calls."""
    checker = run_checker(
        """
from typing import Annotated
def f1() -> Annotated[int, "unit:s"]:
    a: Annotated[int, "unit:s"]
    return a
def f2(a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
    return a
a = f2(f1())
""",
        tmp_path,
    )
    assert_error_u003(checker.errors[0], 8, "test_module.f2", 1, s_unit, m_unit)
    check_unit(checker, "a", m_unit)


def test_expression_function_call_attribute_mismatch(tmp_path: Path):
    """Test  unit mismatches are detected when passing class attributes to functions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:s"]
def f(a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
    return a
a = f(A.a)
""",
        tmp_path,
    )
    assert_error_u003(checker.errors[0], 7, "test_module.f", 1, s_unit, m_unit)


def test_expression_function_call_instance_attribute_mismatch(tmp_path: Path):
    """Test unit mismatches are detected when using instance attributes in functions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:s"]
def f(a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
    return a
a = f(A().a)
""",
        tmp_path,
    )
    assert_error_u003(checker.errors[0], 7, "test_module.f", 1, s_unit, m_unit)


def test_expression_function_args(tmp_path: Path):
    """Test that function arguments with correct units are accepted."""
    checker = run_checker(
        """
from typing import Annotated
def f(a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
    return a
a: Annotated[int, "unit:m"] = 1
b = f(a)
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)
    check_unit(checker, "f", m_unit)
    for node in checker.units:
        assert checker.node_module_names[node] == TEST_MODULE_NAME


def test_expression_method_args(tmp_path: Path):
    """Test that method arguments with correct units are accepted."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def f(self, a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
        return a
a: Annotated[int, "unit:m"] = 1
b = A().f(a)
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_expression_function_args_mismatch(tmp_path: Path):
    """Test that function arguments with incorrect units raise an error."""
    checker = run_checker(
        """
from typing import Annotated
def f(a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
    return a
a: Annotated[int, "unit:s"] = 1
b = f(a)
""",
        tmp_path,
    )
    assert checker.errors, "Expected an error for unit mismatch in function argument"
    assert_error_u003(checker.errors[0], 6, "test_module.f", 1, s_unit, m_unit)


def test_method_args_mismatch(tmp_path: Path):
    """Test that method arguments with incorrect units raise an error."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def f(self, a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
        return a
a: Annotated[int, "unit:s"] = 1
b = A().f(a)
""",
        tmp_path,
    )
    assert_error_u003(checker.errors[0], 7, "test_module.A.f", 1, s_unit, m_unit)


def test_function_no_return_type(tmp_path: Path):
    """Test that functions without a return type annotation are handled gracefully."""
    checker = run_checker(
        """
def f():
    pass
""",
        tmp_path,
    )
    assert not checker.units


def test_function_return_type_no_unit(tmp_path: Path):
    """Test functions with non-unit return types are handled."""
    checker = run_checker(
        """
def f() -> int:
    pass
""",
        tmp_path,
    )
    assert not checker.units


def test_function_return_type_wrong_unit(tmp_path: Path):
    """Test returning a value with the wrong unit from a function raises an error."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> Annotated[int, "unit:m"]:
    a: Annotated[int, "unit:s"]
    return a
""",
        tmp_path,
    )
    assert_error_u004(checker.errors[0], 5, s_unit, m_unit)


def test_function_return_type_wrong_unit_nested(tmp_path: Path):
    """Test that return unit mismatches are detected in nested function definitions."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> Annotated[int, "unit:m"]:
    def f2() -> Annotated[int, "unit:s"]:
        a2: Annotated[int, "unit:m"]
        return a2
    a: Annotated[int, "unit:m"]
    return a
""",
        tmp_path,
    )
    assert_error_u004(checker.errors[0], 6, m_unit, s_unit)


# def test_nested_scope_variable_lookup():
#     checker = run_checker("""
# from typing import Annotated
# a: Annotated[int, "m"]
# def f():
#     b: Annotated[int, "s"]
#     def g():
#         c = a + b
# """)
#     assert_error(
#         checker.errors[0],
#         "U001",
#         "Cannot add operands with different units"
#     )


def test_function_bodies(tmp_path: Path):
    """Test units are tracked within function bodies and errors are reported."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    a: Annotated[int, "unit:m"]
    b: Annotated[int, "unit:s"]
    c = a + b
""",
        tmp_path,
    )
    # check_unit(checker, "f.a", m_unit)
    # check_unit(checker, "f.b", s_unit)
    assert_error_u001(checker.errors[0], 6, m_unit, s_unit)


def test_function_scope_lookup(tmp_path: Path):
    """Test that function scope variable lookup works for return statements."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"]
def f() -> Annotated[int, "unit:s"]:
    return a
""",
        tmp_path,
    )
    assert checker


def test_bare_expression(tmp_path: Path):
    """Test that unit mismatches in expressions are detected and reported."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"]
b: Annotated[int, "unit:s"]
a + b
""",
        tmp_path,
    )
    assert_error_u001(checker.errors[0], 5, m_unit, s_unit)


# def test_function_arg_units(tmp_path: Path):
#     """Test that function arguments are recorded in checker.units."""
#     checker = run_checker(
#         """
# from typing import Annotated
# def f(a: Annotated[int, "m"]):
#     pass
# """,
#         tmp_path,
#     )
#     check_unit(checker, "f.a", m_unit)


def test_class_chaining(tmp_path: Path):
    """Test resolution of expressions with chained class attributes."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
class B:
    A = A
    b: Annotated[int, "unit:m"]
b = B().A.a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)


def test_class_bodies(tmp_path: Path):
    """Test that expressions in class bodies are evaluated."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
    b: Annotated[int, "unit:s"]
    a + b
""",
        tmp_path,
    )
    check_unit(checker, "A.a", m_unit)
    check_unit(checker, "A.b", s_unit)
    assert_error_u001(checker.errors[0], 6, m_unit, s_unit)


def test_closure_type_lookup(tmp_path: Path):
    """Test closure scope lookups of classes."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    class A:
        a: Annotated[int, "unit:m"]
    def f2() -> Annotated[int, "unit:m"]:
        b = A.a
        return b
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "b", m_unit, prefix="")


def test_closure_instance_lookup(tmp_path: Path):
    """Test closure scope lookups of instances."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> None: # annotation required for function variables to be typed
    class A:
        a: Annotated[int, "unit:m"]
    b = A()
    def f2() -> Annotated[int, "unit:m"]:
        c = b.a
        return c
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "c", m_unit, prefix="")


def test_closure_function_lookup(tmp_path: Path):
    """Test closue scope lookup of functi: Pathons."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    def f2() -> Annotated[int, "unit:m"]:
        a: Annotated[int, "unit:m"]
        return a
    def f3() -> Annotated[int, "unit:m"]:
        b = f2()
        return b
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "b", m_unit, prefix="")


def test_closure_variable_lookup(tmp_path: Path):
    """Test closure scope lookup of variables."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    a: Annotated[int, "unit:m"] = 4
    def f2() -> Annotated[int, "unit:m"]:
        b = a
        return b
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "a", m_unit, prefix="")
    check_unit(checker, "b", m_unit, prefix="")


def test_class_nested_scope_variable(tmp_path: Path):
    """Test lookup of variables in arbitrarily nested scopes."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"]
class A:
    b: Annotated[int, "unit:s"]
    def f(self) -> None:
        c = a + self.b
""",
        tmp_path,
    )
    assert_error_u001(checker.errors[0], 7, m_unit, s_unit)


def test_if_else_expr(tmp_path: Path):
    """Test expressions containing if else conditional statements."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"]
b: Annotated[int, "unit:m"]
c: Annotated[int, "unit:s"]
d: int
e: int
f = a if a > b else b  # fine - a and b have same unit
a if a > b else c      # error - a and c have different units
a if a > b else d      # error - a has units, d does not
e if a > b else d      # fine - e and d both have no unit
""",
        tmp_path,
    )
    check_unit(checker, "f", m_unit)
    assert_error_u007(checker.errors[0], m_unit, s_unit, lineno=9)
    assert_error_u008(checker.errors[1], lineno=10)
    assert len(checker.errors) == 2


def test_comparison_expr(tmp_path: Path):
    """Test expressions comparing units with numbers."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"]
b: Annotated[int, "unit:m"]
c: Annotated[int, "unit:s"]
d: int
e: int
a == b # (units match, no error)
b == c # (units differ, error)
a == b == c # (b == c triggers error)
a == d # (unitful and unitless, error)
d == e # (both unitless, no error)
""",
        tmp_path,
    )
    assert_error_u005(checker.errors[0], m_unit, s_unit, lineno=9)
    assert_error_u005(checker.errors[1], m_unit, s_unit, lineno=10)
    assert_error_u006(checker.errors[2], lineno=11)
    assert len(checker.errors) == 3


# def test_module_import():
#     """Test that imported units are recognized."""


#     other_checker = run_checker(
#         """
# from typing import Annotated
# var_with_units: Annotated[int, "m"]
# """,
#         module_name="other_module",
#     )
#     checker = run_checker(
#         """
# from typing import Annotated
# from other_module import var_with_units
# """,
#         external_units=other_checker.units,
#     )
#     assert checker.units["__main__.var_with_units"] == m_unit
#     assert checker.units["other_module.var_with_units"] == m_unit


def test_class_init_attribute(tmp_path: Path):
    """Test that attributes assigned during init statements are handled."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __init__(self):
        self.a: Annotated[int, "unit:m"] = 1
a = A()
b = a.a
c = A().a
""",
        tmp_path,
    )
    check_unit(checker, "A.a", m_unit)
    check_unit(checker, "b", m_unit)
    check_unit(checker, "c", m_unit)


def test_class_self_lookup(tmp_path: Path):
    """Test that accessing variables via self works as expected."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __init__(self):
        self.a: Annotated[int, "unit:m"] = 1
        self.b: Annotated[int, "unit:m"] = 2
        self.c: Annotated[int, "unit:s"] = 3
        self.d = self.a + self.b
        self.b + self.c
""",
        tmp_path,
    )
    check_unit(checker, "A.d", m_unit)
    assert_error_u001(checker.errors[0], 9, m_unit, s_unit)


def test_call_instance(tmp_path: Path):
    """Test that instance __call__ methods are treated correctly."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __call__(self) -> Annotated[int, "unit:m"]:
        a: Annotated[int, "unit:m"]
        return a
a = A()
b = a()
c = A()()
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)
    check_unit(checker, "c", m_unit)


def test_call_instance_error(tmp_path: Path):
    """Test that instance __call__ methods are error checked."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __call__(self, a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
        return a
arg: Annotated[int, "unit:s"] = 1
a = A()
b = a(arg)
c = A()(arg)
""",
        tmp_path,
    )
    assert checker.errors
    assert_error_u003(checker.errors[0], 8, "test_module.A.__call__", 1, s_unit, m_unit)
    assert_error_u003(checker.errors[1], 9, "test_module.A.__call__", 1, s_unit, m_unit)


def test_bin_op_no_units(tmp_path: Path):
    """Checker that binary operations without units work."""
    checker = run_checker(
        """
a = 1
b = 2
a + b""",
        tmp_path,
    )
    assert not checker.errors


def test_bin_op_function(tmp_path: Path):
    """Test binary operations for unusual types."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
def f() -> Annotated[int, "unit:m"]:
    b: Annotated[int, "unit:m"]
    return b
a + f
""",
        tmp_path,
    )
    assert_error_u002(checker.errors[0], 7)


def test_getitem(tmp_path: Path):
    """Test units applied to containers and access of elements."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[list, "unit:m"] = [0, 1]
b = a[0] + a[1]
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "a", m_unit)
    check_unit(checker, "b", m_unit)


def test_getitem_class(tmp_path: Path):
    """Test index expr that returns a class."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"] = 3
b = [A][0].a
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "b", m_unit)


def test_getitem_instance(tmp_path: Path):
    """Test index expr that returns an instance."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"] = 3
b = [A()][0].a
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "b", m_unit)


def test_getitem_function(tmp_path: Path):
    """Test index expr that returns a function."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> Annotated[int, "unit:m"]:
    a: Annotated[int, "unit:m"] = 3
    return a
b = [f][0]()
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "b", m_unit)


def test_function_init_set_self(tmp_path: Path):
    """Test setting units from init functions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __init__(self, a: Annotated[int, "unit:m"]):
        self.a = a
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "A.a", m_unit)


def test_power(tmp_path: Path):
    """Test raising to a power."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[float, "unit:m"] = 3
b = a**2
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "a", m_unit)
    check_unit(checker, "b", m_unit**2)


def test_power_float_error(tmp_path: Path):
    """Test error if non integer value used as exponent."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[float, "unit:m"] = 3
b = a**2.0
""",
        tmp_path,
    )
    assert_error_u009(checker.errors[0], 4)


def test_power_non_int(tmp_path: Path):
    """Test error if non integer value used as exponent."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[float, "unit:m"] = 3
b = 3
c = a**b
""",
        tmp_path,
    )
    assert_error_u009(checker.errors[0], 5)


def test_type_alias_assignment(tmp_path: Path):
    """Check unit annotation via type alias for assignments."""
    checker = run_checker(
        """
from typing import TypeAlias, Annotated, TypeVar
T = TypeVar("T")
metres: TypeAlias = Annotated[T, "unit:m"]
a: metres[int] = 2
""",
        tmp_path,
    )
    ((alias, unit),) = checker.aliases.items()
    assert alias.fullname == "test_module.metres"
    assert unit == m_unit
    check_unit(checker, "a", m_unit)


def test_type_alias_function_return(tmp_path: Path):
    """Check unit annotation via type alias for function return types."""
    checker = run_checker(
        """
from typing import TypeAlias, Annotated, TypeVar
T = TypeVar("T")
metres: TypeAlias = Annotated[T, "unit:m"]
def f() -> metres[int]:
    a: metres[int] = 4
    return a
""",
        tmp_path,
    )
    check_unit(checker, "f", m_unit)


def test_unit_type_alias_function_args(tmp_path: Path):
    """Check unit annotation via type alias for function arguments."""
    checker = run_checker(
        """
from typing import TypeAlias, Annotated, TypeVar
T = TypeVar("T")
metres: TypeAlias = Annotated[T, "unit:m"]
def f(a: metres[int]):
    pass
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "a", m_unit, prefix="")


def test_unit_assignment_error(tmp_path: Path):
    """Check unit errors during assignment."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:s"] = a
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 4, "test_module.b", s_unit, m_unit)


def test_unit_assignment_class_attribute(tmp_path: Path):
    """Test unit errors during assignment to class attributes."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:s"] = 1
A.a = b
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 6, "test_module.A.a", m_unit, s_unit)


def test_unit_assignment_instance_attribute(tmp_path: Path):
    """Test unit errors during assignment to instances."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:s"] = 1
a = A()
a.a = b
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 7, "test_module.A.a", m_unit, s_unit)


def test_unit_assignment_prevent_override(tmp_path: Path):
    """Test preventing overrides using variable annotations."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
a: Annotated[int, "unit:s"] = 2
""",
        tmp_path,
    )
    assert_error_u011(checker.errors[0], 4)


def test_attribute_assignment_prevent_override(tmp_path: Path):
    """Test preventing overrides using variable annotations for class attributes."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
A.a: Annotated[int, "unit:s"]
""",
        tmp_path,
    )
    assert_error_u011(checker.errors[0], 5)


def test_instance_attribute_assignment_prevent_override(tmp_path: Path):
    """Test preventing overrides using variable annotations for class attributes."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
A().a: Annotated[int, "unit:s"]
""",
        tmp_path,
    )
    assert_error_u011(checker.errors[0], 5)


def test_class_set_attribute_error(tmp_path: Path):
    """Check units are respected when setting attributes via method."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"]
    def method(self, a: Annotated[int, "unit:s"]):
        self.a = a
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 6, "test_module.A.a", m_unit, s_unit)


def test_init_argument_error(tmp_path: Path):
    """Check argument errors for __init__ methods."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __init__(self, a: Annotated[int, "unit:m"]):
        pass
b: Annotated[int, "unit:s"]
A(b)
""",
        tmp_path,
    )
    assert_error_u003(checker.errors[0], 7, "test_module.A.__init__", 1, s_unit, m_unit)


def test_unit_retrieval_depth(tmp_path: Path):
    """Check expressions with nested units return unit from correct depth."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "unit:m"] = 1
b: Annotated[A, "unit:s"] = A()
c = b.a
""",
        tmp_path,
    )
    check_unit(checker, "c", m_unit)


def test_getitem_any(tmp_path: Path):
    """Test getitem where underlying type is Any."""
    checker = run_checker(
        """
from typing import Any, Annotated
class A:
    def method(self) -> Annotated[int, "unit:m"]:
        a: Annotated[Any, "unit:m"] = [0, 1]
        b = a[0] - a[1]
        return b
""",
        tmp_path,
    )
    assert not checker.errors


def test_type_alias_return(tmp_path: Path):
    """Test calling a function typed with an alias returns the correct unit."""
    checker = run_checker(
        """
from typing import Annotated, TypeAlias, TypeVar
T = TypeVar("T")
metres: TypeAlias = Annotated[T, "unit:m"]
def f() -> metres[int]:
    a: metres[int] = 1
    return a
a = f()
""",
        tmp_path,
    )
    check_unit(checker, "a", m_unit)


def test_operator_assignment(tmp_path: Path):
    """Test operator assignment statement."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:m"] = 1
a += b
""",
        tmp_path,
    )
    assert not checker.errors


def test_operator_assignment_no_units(tmp_path: Path):
    """Test operator assignment statement with no units on either side."""
    checker = run_checker(
        """
from typing import Annotated
a = 1
b = 1
a += b
""",
        tmp_path,
    )
    assert not checker.errors


def test_operator_assignment_missing_left(tmp_path: Path):
    """Test operator assignment statement with unit missing on left hand side."""
    checker = run_checker(
        """
from typing import Annotated
a = 1
b: Annotated[int, "unit:m"] = 1
a += b
""",
        tmp_path,
    )
    assert_error_u002(checker.errors[0], 5)


def test_operator_assignment_missing_right(tmp_path: Path):
    """Test operator assignment statement with unit missing on right hand side."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b = 1
a += b
""",
        tmp_path,
    )
    assert_error_u002(checker.errors[0], 5)


@pytest.mark.parametrize("operator", ("*", "/"))
def test_operator_assignment_unsupported(tmp_path: Path, operator: str):
    """Check that unsupported operations are not allowed."""
    checker = run_checker(
        f"""
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:s"] = 1
a {operator}= b
""",
        tmp_path,
    )
    assert_error_u012(checker.errors[0], 5, operator)


def test_dictionary_getitem(tmp_path: Path):
    """Test that units are correctly inferred for dictionary values."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[dict, "unit:m"] = {"key1": 1, "key2": 2}
b = a["key1"] + a["key2"]
""",
        tmp_path,
    )
    assert not checker.errors
    check_unit(checker, "b", m_unit)


def test_setitem(tmp_path: Path):
    """Test that units are enforced when setting list items."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[list, "unit:m"] = [1, 2, 3]
b: Annotated[int, "unit:s"] = 4
a[0] = b
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 5, "expression", m_unit, s_unit)


def test_property(tmp_path: Path):
    """Test that property getters provide units."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    @property
    def a(self) -> Annotated[int, "unit:m"]:
        return 1
b = A().a
""",
        tmp_path,
    )
    check_unit(checker, "b", m_unit)
    node = next(node for node in checker.units if node.name == "a")
    assert checker.node_module_names[node] == TEST_MODULE_NAME


def test_property_setter(tmp_path: Path):
    """Test that property setters enforce units."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    _a: Annotated[int, "unit:m"] = 1
    @property
    def a(self) -> Annotated[int, "unit:m"]:
        return self._a
    @a.setter
    def a(self, value: Annotated[int, "unit:m"]):
        self._a = value
obj = A()
obj.a = 2
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 12, "expression", m_unit, None)


def test_assignment_to_existing_variable(tmp_path: Path):
    """Test that re-assigning a variable with a different unit raises an error."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
a = 2
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 4, "test_module.a", m_unit, None)


def test_assignment_to_existing_with_annotation(tmp_path: Path):
    """Test that annotated units only accept unitless quantities on first definition."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "unit:m"] = 1
a: Annotated[int, "unit:m"] = 2
""",
        tmp_path,
    )
    assert_error_u010(checker.errors[0], 4, "test_module.a", m_unit, None)


def test_overloaded_function_definition_not_implemented(tmp_path: Path):
    """Overloaded functions are not supported."""
    with pytest.raises(NotImplementedError):
        run_checker(
            """
from typing import Annotated, overload
class A:
    @overload
    def a(self, a: int) -> None:
        pass
    @overload
    def a(self, a: str) -> None:
        pass
""",
            tmp_path,
        )
