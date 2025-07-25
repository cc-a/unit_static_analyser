"""UnitChecker module."""

import ast
from ..units.units import Unit

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
        self.units: dict[str, Unit] = {}
        self.errors: list[UnitCheckerError] = []
        self.scope_stack: list[str] = []

    def scoped_key(self, var_name: str) -> str:
        if self.scope_stack:
            return ".".join(self.scope_stack + [var_name])
        return var_name

    def lookup_unit(self, var_name: str) -> Unit | None:
        # Try all possible scope prefixes, from innermost to outermost
        for i in range(len(self.scope_stack), -1, -1):
            scope = self.scope_stack[:i]
            key = ".".join(scope + [var_name]) if scope else var_name
            if key in self.units:
                return self.units[key]
        return None

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            unit_str = self._extract_unit_from_annotation(node.annotation)
            if unit_str is not None:
                self.units[self.scoped_key(var_name)] = Unit.from_string(unit_str)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to check unit operations and compatibility."""
        if isinstance(node.value, ast.BinOp):
            left_var = node.value.left
            right_var = node.value.right
            op = node.value.op

            if not (isinstance(node.targets[0], ast.Name)):
                return

            target = node.targets[0].id

            left_unit = self.lookup_unit(left_var.id) if isinstance(left_var, ast.Name) else None
            right_unit = self.lookup_unit(right_var.id) if isinstance(right_var, ast.Name) else None

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
                self.units[self.scoped_key(target)] = left_unit
            elif isinstance(op, ast.Sub):
                if left_unit != right_unit:
                    self.errors.append(
                        UnitCheckerError(
                            code="U001",
                            lineno=node.lineno,
                            message=f"Cannot subtract operands with different units: {left_unit} and {right_unit}",
                        )
                    )
                    return
                self.units[self.scoped_key(target)] = left_unit
            elif isinstance(op, ast.Mult):
                result_unit = left_unit * right_unit
                self.units[self.scoped_key(target)] = result_unit
            elif isinstance(op, ast.Div):
                result_unit = left_unit * (right_unit ** -1)
                self.units[self.scoped_key(target)] = result_unit
            elif isinstance(op, ast.Pow):
                if isinstance(node.value.right, ast.Constant) and isinstance(node.value.right.value, int):
                    result_unit = left_unit ** node.value.right.value
                    self.units[self.scoped_key(target)] = result_unit
                else:
                    self.errors.append(
                        UnitCheckerError(
                            code="U003",
                            lineno=node.lineno,
                            message="Exponent must be an integer constant",
                        )
                    )
            else:
                self.errors.append(
                    UnitCheckerError(
                        code="U004",
                        lineno=node.lineno,
                        message=f"Unsupported operation for units: {type(op).__name__}",
                    )
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _extract_unit_from_annotation(self, annotation: ast.AST) -> str | None:
        # Extract unit symbol from typing.Annotated[<type>, <unit>]
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
                if isinstance(slice_value, ast.Tuple) and len(slice_value.elts) > 1:
                    unit_elt = slice_value.elts[1]
                    if isinstance(unit_elt, ast.Constant) and isinstance(unit_elt.value, str):
                        return unit_elt.value  # Extract string literal
        return None
