import abc
import ast
import pytest

from unit_static_analyser.mypy_checker.checker import UnitChecker
from unit_static_analyser.units import Unit

from mypy.nodes import TypeInfo

# Define unit instances at module scope
m_unit = Unit.from_string("m")
from typing import Annotated

s_unit = Unit.from_string("s")
kg_unit = Unit.from_string("kg")


from unit_static_analyser.mypy_checker.checker import UnitChecker


def run_checker(code: str, module_name="__main__", external_units=None) -> UnitChecker:
    checker = UnitChecker(module_name=module_name)
    checker.check(code)
    if external_units:
        checker.units.update(external_units)
    return checker


def assert_error(error, code, msg_contains=None, lineno=None):
    assert error.code == code
    if lineno:
        assert error.lineno == lineno
    if msg_contains:
        assert msg_contains in error.message


def test_assignment_arbitrary_expression():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
def f() -> A:
    return A()
b = f().a
""")
    assert checker.units["__main__.b"] == m_unit


def test_assignment_arbitrary_expression2():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
def f() -> A:
    return A
b = f().a
""")
    assert checker.units["__main__.b"] == m_unit


def test_assignment_arbitrary_expression3():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
def f() -> A:
    return A
def f2(a: Annotated[int, "s"]) -> Annotated[int, "s"]:
    return a
b = f2(f().a)
""")
    assert checker.units["__main__.b"] == s_unit
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f2'")
    assert "has unit m, expected s" in checker.errors[0].message


def test_assignment_instance_attribute():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
a = A()
b = a.a
c = A().a
""")
    assert checker.units["__main__.c"] == m_unit
    assert checker.units["__main__.b"] == m_unit


def test_assignment_member_access():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
b = A.a
""")
    assert checker.units["__main__.b"] == m_unit


def test_assignment_function_call():
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    a: Annotated[int, "m"]
b = f()
""")
    assert checker.units["__main__.b"] == m_unit


def test_assignment_class_init_member_access():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"]
b = A().a
""")
    assert checker.units["__main__.b"] == m_unit


def test_assignment_alias():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
b = a
""")
    assert checker.units["__main__.b"] == m_unit


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


@pytest.mark.parametrize("symbol", ("+", "-"))
def test_expression_unary(symbol):
    checker = run_checker(f"""
from typing import Annotated
a: Annotated[int, "m"] = 1
b = {symbol}a
""")
    assert checker.units["__main__.b"] == m_unit


def test_expression_instance_method():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "m"] = 1
    def f(self) -> Annotated[int, "m"]:
        return self.a
b = A().f()
""")
    assert checker.units["__main__.b"] == m_unit


def test_expression_nested_function_call_mismatch():
    checker = run_checker("""
from typing import Annotated
def f1() -> Annotated[int, "s"]:
    a: Annotated[int, "s"]
    return a
def f2(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a = f2(f1())
""")
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f2'")
    assert checker.units["__main__.a"] == m_unit


def test_expression_function_call_attribute_mismatch():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "s"]
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a = f(A.a)
""")
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f'")


def test_expression_function_call_instance_attribute_mismatch():
    checker = run_checker("""
from typing import Annotated
class A:
    a: Annotated[int, "s"]
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a = f(A().a)
""")
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f'")


def test_expression_function_args():
    checker = run_checker("""
from typing import Annotated
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a: Annotated[int, "m"] = 1
b = f(a)
""")
    assert checker.units["__main__.b"] == m_unit
    assert checker.function_returns["__main__.f"] == m_unit


def test_expression_function_args_mismatch():
    checker = run_checker("""
from typing import Annotated
def f(a: Annotated[int, "m"]) -> Annotated[int, "m"]:
    return a
a: Annotated[int, "s"] = 1
b = f(a)
""")
    assert checker.errors, "Expected an error for unit mismatch in function argument"
    assert_error(checker.errors[0], "U003", "Argument 1 to function 'f'")
    assert "has unit s, expected m" in checker.errors[0].message


def test_function_no_return_type():
    checker = run_checker("""
from typing import Annotated
def f():
    pass
""")
    assert "__main__.f" not in checker.function_returns


def test_function_return_type_no_unit():
    checker = run_checker("""
from typing import Annotated
def f() -> int:
    pass
""")

    ret_value = checker.function_returns["__main__.f"]
    assert isinstance(ret_value, TypeInfo)
    assert ret_value.fullname == "builtins.int"


def test_function_return_type_wrong_unit():
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    a: Annotated[int, "s"]
    return a
""")
    assert_error(
        checker.errors[0],
        "U004",
        "Unit of return value does not match function signature",
    )


def test_function_return_type_wrong_unit_nested():
    checker = run_checker("""
from typing import Annotated
def f() -> Annotated[int, "m"]:
    def f2() -> Annotated[int, "s"]:
        a2: Annotated[int, "m"]
        return a2
    a: Annotated[int, "m"]
    return a
""")
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
#     assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_function_bodies():
    checker = run_checker("""
from typing import Annotated
def f():
    a: Annotated[int, "m"]
    b: Annotated[int, "s"]
    c = a + b
""")
    assert checker.units["__main__.f.a"] == m_unit
    assert checker.units["__main__.f.b"] == s_unit
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


def test_function_scope_lookup():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
def f() -> Annotated[int, "s"]:
    return a
""")
    assert checker


def test_expression():
    checker = run_checker("""
from typing import Annotated
a: Annotated[int, "m"]
b: Annotated[int, "s"]
a + b
""")
    assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


# def test_class_bodies():
#     checker = run_checker("""
# from typing import Annotated
# class A:
#     a: Annotated[int, "m"]
#     b: Annotated[int, "s"]
#     a + b
# """)
#     assert checker.units["__main__.A.a"] == m_unit
#     assert checker.units["__main__.A.b"] == s_unit
#     assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


# def test_class_nested_scope_variable():
#     checker = run_checker("""
# from typing import Annotated
# a: Annotated[int, "m"]
# class A:
#     b: Annotated[int, "s"]
#     def f(self):
#         c = a + b
# """)
#     assert_error(checker.errors[0], "U001", "Cannot add operands with different units")


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
