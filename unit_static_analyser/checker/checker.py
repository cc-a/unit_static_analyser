"""UnitChecker module."""

import ast

from ..units import types as units_mod


class UnitChecker(ast.NodeVisitor):
    """A static analysis visitor to check unit compatibility in Python code."""

    def __init__(self) -> None:
        """Initialize the UnitChecker with an empty unit mapping."""
        self.units: dict[str, type] = {}

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignments to extract unit information."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            unit_cls = self._extract_unit_class_from_annotation(node.annotation)
            if unit_cls is not None:
                self.units[var_name] = unit_cls

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignments to check unit operations and compatibility."""
        if isinstance(node.value, ast.BinOp):
            left_var = node.value.left
            right_var = node.value.right
            op = node.value.op
            if (
                isinstance(left_var, ast.Name)
                and isinstance(right_var, ast.Name)
                and isinstance(node.targets[0], ast.Name)
            ):
                left = left_var.id
                right = right_var.id
                target = node.targets[0].id
                left_unit = self.units.get(left)
                right_unit = self.units.get(right)
                if isinstance(op, ast.Add):
                    if (
                        left_unit is not None
                        and right_unit is not None
                        and left_unit != right_unit
                    ):
                        raise TypeError(
                            f"Cannot add {left} ({left_unit}) and {right}"
                            f" ({right_unit})"
                        )
                    if left_unit is not None:
                        self.units[target] = left_unit
                elif isinstance(op, ast.Div):
                    if left_unit is not None and right_unit is not None:
                        result_unit = left_unit / right_unit  # type: ignore
                        self.units[target] = result_unit

    def _extract_unit_class_from_annotation(self, annotation: ast.AST) -> type | None:
        # Similar to previous version, updated for new import
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
            if ann_id == "Quantity":
                slice_value = annotation.slice
                if isinstance(slice_value, ast.Tuple) and len(slice_value.elts) > 1:
                    unit_elt = slice_value.elts[1]
                    if isinstance(unit_elt, ast.Name):
                        return getattr(units_mod, unit_elt.id, None)
                    if isinstance(unit_elt, ast.Call):
                        if (
                            isinstance(unit_elt.func, ast.Name)
                            and unit_elt.func.id == "type"
                        ):
                            expr = unit_elt.args[0]
                            code = compile(ast.Expression(expr), "<string>", "eval")
                            return eval(code, vars(units_mod))
        return None
