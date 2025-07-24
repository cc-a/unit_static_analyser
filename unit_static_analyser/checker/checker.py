"""UnitChecker module."""

import ast

from .. import units as units_mod


class UnitCheckerError:
    """Represents a unit checking error."""

    def __init__(self, code: str, lineno: int, message: str):
        self.code = code
        self.lineno = lineno
        self.message = message

    def __repr__(self):
        return f"UnitCheckerError(code={self.code!r}, lineno={self.lineno!r}, message={self.message!r})"


class UnitChecker(ast.NodeVisitor):
    """A static analysis visitor to check unit compatibility in Python code."""

    def __init__(self) -> None:
        """Initialize the UnitChecker with an empty unit mapping and error list."""
        self.units: dict[str, object] = {}
        self.errors: list[UnitCheckerError] = []

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignments to extract unit information."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            unit_obj = self._extract_unit_from_annotation(node.annotation)
            if unit_obj is not None:
                self.units[var_name] = unit_obj

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to check unit operations and compatibility."""
        if isinstance(node.value, ast.BinOp):
            left_var = node.value.left
            right_var = node.value.right
            op = node.value.op

            # Only handle assignments to a variable
            if not (isinstance(node.targets[0], ast.Name)):
                return

            target = node.targets[0].id

            # Determine left unit
            if isinstance(left_var, ast.Name):
                left = left_var.id
                left_unit = self.units.get(left)
            else:
                left_unit = None
            # Determine right unit
            if isinstance(right_var, ast.Name):
                right = right_var.id
                right_unit = self.units.get(right)
            else:
                right_unit = None

            # If either operand is missing a unit, collect error
            if left_unit is None or right_unit is None:
                self.errors.append(
                    UnitCheckerError(
                        code="U002",
                        lineno=node.lineno,
                        message="Operands must both have units",
                    )
                )
                return

            if isinstance(op, ast.Add):
                if left_unit != right_unit:
                    self.errors.append(
                        UnitCheckerError(
                            code="U001",
                            lineno=node.lineno,
                            message=f"Cannot add operands with different units: {left_unit} and {right_unit}",
                        )
                    )
                    return
                self.units[target] = left_unit
            elif isinstance(op, ast.Div):
                result_unit = left_unit / right_unit
                self.units[target] = result_unit

    def _extract_unit_from_annotation(self, annotation: ast.AST) -> object | None:
        # Extract unit instance from typing.Annotated[<type>, <unit>]
        if isinstance(annotation, ast.Subscript):
            ann_id = (
                annotation.value.id
                if isinstance(annotation.value, ast.Name)
                else (
                    annotation.value.attr
                    if isinstance(annotation.value, ast.Attribute)
                    else None
                )
            )
            if ann_id == "Annotated":
                slice_value = annotation.slice
                # For Python 3.9+, slice is an ast.Tuple; for older, may be ast.Index
                if isinstance(slice_value, ast.Tuple) and len(slice_value.elts) > 1:
                    unit_elt = slice_value.elts[1]
                    # Evaluate the unit instance from the annotation
                    if isinstance(unit_elt, ast.Name):
                        return getattr(units_mod, unit_elt.id, None)
                    if isinstance(unit_elt, ast.Call):
                        # Support e.g. Annotated[int, m**2]
                        code = compile(ast.Expression(unit_elt), "<string>", "eval")
                        return eval(code, vars(units_mod))
        return None
