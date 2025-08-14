import pytest
from mypy.nodes import TypeInfo

from unit_static_analyser.mypy_checker.checker import FuncUnitDescription, UnitChecker
from unit_static_analyser.units import Unit

# Define unit instances at module scope
m_unit = Unit.from_string("m")
s_unit = Unit.from_string("s")
kg_unit = Unit.from_string("kg")


def run_checker(
    code: str, tmp_path, module_name="__main__", external_units=None
) -> UnitChecker:
    """Run the UnitChecker on the given code using a temp file via pytest's tmp_path.

    Args:
        code: The Python code to analyze.
        tmp_path: pytest's temporary directory fixture.
        module_name: The module name to use for the analysis.
        external_units: Optional dictionary of external units to include.

    Returns:
        The UnitChecker instance after analysis.
    """
    file_path = tmp_path / "test_module.py"
    file_path.write_text(code)
    checker = UnitChecker(module_name=module_name)
    checker.check([str(file_path)])
    if external_units:
        checker.units.update(external_units)
    return checker


def assert_error(error, code, msg_contains=None, lineno=None):
    """Assert that an error matches the expected code, message content, and line number.

    Args:
        error: The UnitCheckerError instance to check.
        code: The expected error code.
        msg_contains: Optional substring that should be in the error message.
        lineno: Optional expected line number for the error.
    """
    assert error.code == code
    if lineno:
        assert error.lineno == lineno
    if msg_contains:
        assert msg_contains in error.message


def assert_error_u005(error, left_unit, right_unit, lineno=None):
    """Assert that a U005 error matches expected left/right units and message."""
    assert error.code == "U005"
    expected_msg = (
        f"Cannot compare operands with different units: {left_unit} and {right_unit}"
    )
    assert expected_msg in error.message
    if lineno is not None:
        assert error.lineno == lineno


def assert_error_u006(error, lineno=None):
    """Assert that a U006 error matches expected left/right units and message."""
    assert error.code == "U006"
    expected_msg = "Cannot compare a unitful operand with a unitless operand"
    assert expected_msg in error.message
    if lineno is not None:
        assert error.lineno == lineno


def assert_error_u007(error, if_unit, else_unit, lineno=None):
    """Assert that a U007 error matches expected if/else units and message."""
    assert error.code == "U007"
    expected_msg = (
        f"Conditional branches have different units: {if_unit} and {else_unit}"
    )
    assert expected_msg in error.message
    if lineno is not None:
        assert error.lineno == lineno


def assert_error_u008(error, lineno=None):
    """Assert that a U008 error matches expected message."""
    assert error.code == "U008"
    expected_msg = "Both branches of conditional must a unit."
    assert expected_msg in error.message
    if lineno is not None:
        assert error.lineno == lineno


def test_assignment_arbitrary_expression(tmp_path):
    """Test that units are correctly inferred for arbitrary complex expressions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
def f() -> A:
    return A()
b = f().a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_assignment_arbitrary_expression2(tmp_path):
    """Another test that units are inferred for arbitrary complex expressions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
def f() -> A:
    return A
b = f().a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_assignment_arbitrary_expression3(tmp_path):
    """Another test that units are inferred for arbitrary complex expressions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
def f() -> A:
    return A
def f2(a: Annotated[int, "s"]) -> Annotated[int, "s"]:
    return a
b = f2(f().a)
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == s_unit
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f2'")
    assert "has unit m, expected s" in checker.errors[0].message


def test_function_return_instance(tmp_path):
    """Test unit lookups when operating on a object returned by a function."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
    def f(self, b: Annotated[int, "m"]) -> Annotated[int, "m"]:
        return a
def f() -> A:
    return A()
b: Annotated[int, "s"] = 1
c = f().f(b)
d = f().a
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'A.f'")
    assert "has unit s, expected m" in checker.errors[0].message
    assert checker.units["__main__.c"] == m_unit
    assert checker.units["__main__.d"] == m_unit


def test_assignment_instance_attribute(tmp_path):
    """Test that units are correctly inferred for instance attribute access."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
a = A()
b = a.a
c = A().a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit
    assert checker.units["__main__.c"] == m_unit


def test_assignment_member_access(tmp_path):
    """Test that units are correctly inferred for class attribute access."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
b = A.a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_assignment_function_call(tmp_path):
    """Test units are inferred for function return values with unit annotations."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> Annotated[int, "m"]:
    a: Annotated[int, "m"]
b = f()
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_assignment_class_init_member_access(tmp_path):
    """Test units are inferred for attribute access on instances created inline."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
b = A().a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_assignment_alias(tmp_path):
    """Test that units are correctly propagated through variable aliasing."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"]
b = a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_assignment(tmp_path):
    """Test that units are correctly assigned to annotated variables."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"] = 1
""",
        tmp_path,
    )
    assert checker.units["__main__.a"] == m_unit


def test_addition(tmp_path):
    """Test addition of variables with same unit is allowed and unit is preserved."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"] = 1
b: Annotated[int, "m"] = 2
c = a + b
""",
        tmp_path,
    )
    for name in ("a", "b", "c"):
        assert checker.units[f"__main__.{name}"] == m_unit
    assert not checker.errors


def test_addition_error(tmp_path):
    """Test that addition of variables with different units raises an error."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"] = 1
b: Annotated[int, "s"] = 2
c = a + b
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_division_ok(tmp_path):
    """Test division of variables with units produces the correct resulting unit."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"] = 1
b: Annotated[int, "s"] = 2
c = a / b
""",
        tmp_path,
    )
    assert checker.units["__main__.c"] == Unit.from_string("m.s^-1")


def test_disallow_missing_units(tmp_path):
    """Test that operations involving unannotated values raise an error."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"] = 1
b = a + 4
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U002", "Operands must both have units")


@pytest.mark.parametrize("symbol", ("+", "-"))
def test_expression_unary(symbol, tmp_path):
    """Test that unary plus and minus preserve the unit of the operand."""
    checker = run_checker(
        f"""
from typing import Annotated
a: Annotated[int, "m"] = 1
b = {symbol}a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_expression_instance_method(tmp_path):
    """Test that units are correctly inferred for return values of instance methods."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"] = 1
    def f(self) -> Annotated[int, "m"]:
        return self.a
b = A().f()
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_expression_nested_function_call_mismatch(tmp_path):
    """Test that unit mismatches are detected in nested function calls."""
    checker = run_checker(
        """
from typing import Annotated
def f1() -> Annotated[int, "s"]:
    a: Annotated[int, "s"]
    return a
def f2(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a = f2(f1())
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f2'")
    assert checker.units["__main__.a"] == m_unit


def test_expression_function_call_attribute_mismatch(tmp_path):
    """Test  unit mismatches are detected when passing class attributes to functions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "s"]
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a = f(A.a)
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f'")


def test_expression_function_call_instance_attribute_mismatch(tmp_path):
    """Test unit mismatches are detected when using instance attributes in functions."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "s"]
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a = f(A().a)
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f'")


def test_expression_function_args(tmp_path):
    """Test that function arguments with correct units are accepted."""
    checker = run_checker(
        """
from typing import Annotated
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a: Annotated[int, "m"] = 1
b = f(a)
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit
    func_units = checker.function_units["__main__.f"]
    assert isinstance(func_units.returns, Unit)
    assert func_units.returns == m_unit


def test_expression_method_args(tmp_path):
    """Test that method arguments with correct units are accepted."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def f(self, a: Annotated[int, "m"]) -> Annotated[int, "m"]:
        return a
a: Annotated[int, "m"] = 1
b = A().f(a)
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_expression_function_args_mismatch(tmp_path):
    """Test that function arguments with incorrect units raise an error."""
    checker = run_checker(
        """
from typing import Annotated
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a: Annotated[int, "s"] = 1
b = f(a)
""",
        tmp_path,
    )
    assert checker.errors, "Expected an error for unit mismatch in function argument"
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f'")
    assert "has unit s, expected m" in checker.errors[0].message


def test_method_args_mismatch(tmp_path):
    """Test that method arguments with incorrect units raise an error."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def f(self, a: Annotated[int, "m"]) -> Annotated[int, "m"]:
        return a
a: Annotated[int, "s"] = 1
b = A().f(a)
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'A.f'")
    assert "has unit s, expected m" in checker.errors[0].message


def test_function_no_return_type(tmp_path):
    """Test that functions without a return type annotation are handled gracefully."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    pass
""",
        tmp_path,
    )
    assert "__main__.f" not in checker.function_units


def test_function_return_type_no_unit(tmp_path):
    """Test functions with non-unit return types are handled."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> int:
    pass
""",
        tmp_path,
    )

    ret_value = checker.function_units["__main__.f"]
    assert isinstance(ret_value, FuncUnitDescription)
    assert ret_value.returns
    assert isinstance(ret_value.returns, TypeInfo)
    assert ret_value.returns.fullname == "builtins.int"


def test_function_return_type_wrong_unit(tmp_path):
    """Test returning a value with the wrong unit from a function raises an error."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> Annotated[int, "m"]:
    a: Annotated[int, "s"]
    return a
""",
        tmp_path,
    )
    assert_error(
        checker.errors[0],
        "U004",
        "Unit of return value does not match function signature",
    )


def test_function_return_type_wrong_unit_nested(tmp_path):
    """Test that return unit mismatches are detected in nested function definitions."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> Annotated[int, "m"]:
    def f2() -> Annotated[int, "s"]:
        a2: Annotated[int, "m"]
        return a2
    a: Annotated[int, "m"]
    return a
""",
        tmp_path,
    )
    assert_error(
        checker.errors[0],
        "U004",
        "Unit of return value does not match function signature",
        lineno=6,
    )


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


def test_function_bodies(tmp_path):
    """Test units are tracked within function bodies and errors are reported."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    a: Annotated[int, "m"]
    b: Annotated[int, "s"]
    c = a + b
""",
        tmp_path,
    )
    assert checker.units["__main__.f.a"] == m_unit
    assert checker.units["__main__.f.b"] == s_unit
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_function_scope_lookup(tmp_path):
    """Test that function scope variable lookup works for return statements."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"]
def f() -> Annotated[int, "s"]:
    return a
""",
        tmp_path,
    )
    assert checker


def test_bare_expression(tmp_path):
    """Test that unit mismatches in expressions are detected and reported."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"]
b: Annotated[int, "s"]
a + b
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_function_arg_units(tmp_path):
    """Test that function arguments are recorded in checker.units."""
    checker = run_checker(
        """
from typing import Annotated
def f(a: Annotated[int, "m"]):
    pass
""",
        tmp_path,
    )
    assert checker.units["__main__.f.a"] == m_unit


def test_class_chaining(tmp_path):
    """Test resolution of expressions with chained class attributes."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
class B:
    A = A
    b: Annotated[int, "m"]
b = B().A.a
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit


def test_class_bodies(tmp_path):
    """Test that expressions in class bodies are evaluated."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    a: Annotated[int, "m"]
    b: Annotated[int, "s"]
    a + b
""",
        tmp_path,
    )
    assert checker.units["__main__.A.a"] == m_unit
    assert checker.units["__main__.A.b"] == s_unit
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_closure_type_lookup(tmp_path):
    """Test closure scope lookups of classes."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    class A:
        a: Annotated[int, "m"]
    def f2() -> Annotated[int, "m"]:
        b = A.a
        return b
""",
        tmp_path,
    )
    assert not checker.errors
    assert checker.units["__main__.f.f2.b"] == m_unit


def test_closure_instance_lookup(tmp_path):
    """Test closure scope lookups of instances."""
    checker = run_checker(
        """
from typing import Annotated
def f() -> None: # annotation required for function variables to be typed
    class A:
        a: Annotated[int, "m"]
    b = A()
    def f2() -> Annotated[int, "m"]:
        c = b.a
        return c
""",
        tmp_path,
    )
    assert not checker.errors
    assert checker.units["__main__.f.f2.c"] == m_unit


def test_closure_function_lookup(tmp_path):
    """Test closue scope lookup of functions."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    def f2() -> Annotated[int, "m"]:
        a: Annotated[int, "m"]
        return a
    def f3() -> Annotated[int, "m"]:
        b = f2()
        return b
""",
        tmp_path,
    )
    assert not checker.errors
    assert checker.units["__main__.f.f3.b"] == m_unit


def test_closure_variable_lookup(tmp_path):
    """Test closure scope lookup of variables."""
    checker = run_checker(
        """
from typing import Annotated
def f():
    a: Annotated[int, "m"] = 4
    def f2() -> Annotated[int, "m"]:
        b = a
        return b
""",
        tmp_path,
    )
    assert not checker.errors
    assert checker.units["__main__.f.a"] == m_unit
    assert checker.units["__main__.f.f2.b"] == m_unit


def test_class_nested_scope_variable(tmp_path):
    """Test lookup of variables in arbitrarily nested scopes."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"]
class A:
    b: Annotated[int, "s"]
    def f(self):
        c = a + b
""",
        tmp_path,
    )
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_if_else_expr(tmp_path):
    """Test expressions containing if else conditional statements."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"]
b: Annotated[int, "m"]
c: Annotated[int, "s"]
d: int
e: int
f = a if a > b else b  # fine - a and b have same unit
a if a > b else c      # error - a and c have different units
a if a > b else d      # error - a has units, d does not
e if a > b else d      # fine - e and d both have no unit
""",
        tmp_path,
    )
    assert checker.units["__main__.f"] == m_unit
    assert_error_u007(checker.errors[0], m_unit, s_unit, lineno=9)
    assert_error_u008(checker.errors[1], lineno=10)
    assert len(checker.errors) == 2


def test_comparison_expr(tmp_path):
    """Test expressions comparing units with numbers."""
    checker = run_checker(
        """
from typing import Annotated
a: Annotated[int, "m"]
b: Annotated[int, "m"]
c: Annotated[int, "s"]
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


def test_class_init_attribute(tmp_path):
    """Test that attributes assigned during init statements are handled."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __init__(self):
        self.a: Annotated[int, "m"] = 1
a = A()
b = a.a
c = A().a
""",
        tmp_path,
    )
    assert checker.units["__main__.A.a"] == m_unit
    assert checker.units["__main__.b"] == m_unit
    assert checker.units["__main__.c"] == m_unit


def test_class_self_lookup(tmp_path):
    """Test that accessing variables via self works as expected."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __init__(self):
        self.a: Annotated[int, "m"] = 1
        self.b: Annotated[int, "m"] = 2
        self.c: Annotated[int, "s"] = 3
        self.d = self.a + self.b
        self.b + self.c
""",
        tmp_path,
    )
    assert checker.units["__main__.A.d"] == m_unit
    assert_error(
        checker.errors[0], "U001", "Cannot add operands with different units", lineno=9
    )


def test_call_instance(tmp_path):
    """Test that instance __call__ methods are treated correctly."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __call__(self) -> Annotated[int, "m"]:
        a: Annotated[int, "m"]
        return a
a = A()
b = a()
c = A()()
""",
        tmp_path,
    )
    assert checker.units["__main__.b"] == m_unit
    assert checker.units["__main__.c"] == m_unit


def test_call_instance_error(tmp_path):
    """Test that instance __call__ methods are error checked."""
    checker = run_checker(
        """
from typing import Annotated
class A:
    def __call__(self, a: Annotated[int, "m"]) -> Annotated[int, "m"]:
        return a
arg: Annotated[int, "s"] = 1
a = A()
b = a(arg)
c = A()(arg)
""",
        tmp_path,
    )
    assert checker.errors
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'A.__call__'")
    assert "has unit s, expected m" in checker.errors[0].message
