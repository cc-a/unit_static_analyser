"""Module for creating errors representing invalid unit operations."""

from ..units import Unit


class UnitCheckerError:
    """Represents a unit checking error."""

    def __init__(self, code: str, lineno: int, message: str):
        """Initialise a new unit checking error."""
        self.code = code
        self.lineno = lineno
        self.message = message

    def __repr__(self) -> str:
        """Return a string representation of the error."""
        return (
            "UnitCheckerError"
            f"(code={self.code!r}, lineno={self.lineno!r}, message={self.message!r})"
        )


def u001_error_factory(
    lineno: int, left_unit: Unit, right_unit: Unit
) -> UnitCheckerError:
    """Factory for U001: Cannot add operands with different units."""
    return UnitCheckerError(
        code="U001",
        lineno=lineno,
        message=(
            f"Cannot add operands with different units: {left_unit} and {right_unit}"
        ),
    )


def u002_error_factory(lineno: int) -> UnitCheckerError:
    """Factory for U002: Operands must both have units."""
    return UnitCheckerError(
        code="U002",
        lineno=lineno,
        message="Operands must both have units",
    )


def u003_error_factory(
    lineno: int,
    arg_index: int,
    func_fullname: str,
    inferred_unit: Unit | None,
    expected_unit: Unit,
) -> UnitCheckerError:
    """Factory for U003: Argument to function has wrong unit."""
    return UnitCheckerError(
        code="U003",
        lineno=lineno,
        message=(
            f"Argument {arg_index} to function '{func_fullname}' "
            f"has unit {inferred_unit}, expected {expected_unit}"
        ),
    )


def u004_error_factory(
    lineno: int, returned_unit: Unit | None, return_unit: Unit
) -> UnitCheckerError:
    """Factory for U004: Unit of return value does not match function signature."""
    return UnitCheckerError(
        code="U004",
        lineno=lineno,
        message=(
            "Unit of return value does not match function "
            f"signature: returned {returned_unit}, "
            f"expected {return_unit}"
        ),
    )


def u005_error_factory(
    lineno: int, left_unit: Unit | None, right_unit: Unit | None
) -> UnitCheckerError:
    """Factory for U005: Cannot compare operands with different units."""
    return UnitCheckerError(
        code="U005",
        lineno=lineno,
        message=(
            f"Cannot compare operands with different units: "
            f"{left_unit} and {right_unit}"
        ),
    )


def u006_error_factory(lineno: int) -> UnitCheckerError:
    """Factory for U006: Cannot compare a unitful operand with a unitless operand."""
    return UnitCheckerError(
        code="U006",
        lineno=lineno,
        message="Cannot compare a unitful operand with a unitless operand",
    )


def u007_error_factory(
    lineno: int, true_unit: Unit, false_unit: Unit
) -> UnitCheckerError:
    """Factory for U007: Conditional branches have different units."""
    return UnitCheckerError(
        code="U007",
        lineno=lineno,
        message=(
            f"Conditional branches have different units: {true_unit} and {false_unit}"
        ),
    )


def u008_error_factory(lineno: int) -> UnitCheckerError:
    """Factory for U008: Both branches of conditional must have a unit."""
    return UnitCheckerError(
        code="U008",
        lineno=lineno,
        message="Both branches of conditional must a unit.",
    )


def u009_error_factory(lineno: int) -> UnitCheckerError:
    """Factory for U009: Exponent must be an explicit integer value."""
    return UnitCheckerError(
        code="U009",
        lineno=lineno,
        message="Exponent must be an explicit integer value.",
    )


def u010_error_factory(
    lineno: int, fullname: str, expected_unit: Unit, inferred_unit: Unit
) -> UnitCheckerError:
    """Factory for U010: Incompatible unit in assignment."""
    return UnitCheckerError(
        code="U010",
        lineno=lineno,
        message=(
            f"Incompatible unit in assignment to {fullname}: expected {expected_unit}, "
            f"received {inferred_unit}"
        ),
    )


def u011_error_factory(lineno: int) -> UnitCheckerError:
    """Factory for U011: Variable already has a unit."""
    return UnitCheckerError(
        code="U011",
        lineno=lineno,
        message="Variable already has a unit",
    )
