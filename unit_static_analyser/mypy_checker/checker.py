"""UnitChecker using mypy for type analysis.

This module provides a static analysis tool for extracting and checking physical units
from Python code using type annotations and mypy's typed AST.
"""

import os
import tempfile

from mypy import build
from mypy.nodes import (
    AssignmentStmt,
    CallExpr,
    ClassDef,
    Expression,
    ExpressionStmt,
    FuncDef,
    MemberExpr,
    MypyFile,
    NameExpr,
    OpExpr,
    ReturnStmt,
    Statement,
    TypeInfo,
    UnaryExpr,
    Var,
)
from mypy.options import Options
from mypy.types import CallableType, Instance, UnboundType, get_proper_type

from unit_static_analyser.units import Unit


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


class UnitChecker:
    """Uses mypy's typed AST to extract and check units."""

    def __init__(self, module_name: str = "__main__") -> None:
        """Initialise a new checker.

        Args:
            module_name: module of interest
        """
        self.units: dict[str, Unit] = {}
        self.errors: list[UnitCheckerError] = []
        self.module_name = module_name
        self.function_returns: dict[str, Unit | TypeInfo] = {}

    def check(self, code: str) -> None:
        """Run mypy on the code and extract units from Annotated assignments."""
        # Write code to a temporary file so mypy does full type inference
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_filename = tmp.name

        options = Options()
        options.incremental = False
        options.show_traceback = True
        options.namespace_packages = True
        options.ignore_missing_imports = True
        options.follow_imports = "silent"
        options.allow_untyped_globals = True
        options.check_untyped_defs = False
        options.use_builtins_fixtures = True
        options.export_types = True
        options.preserve_asts = True

        result = build.build(
            sources=[build.BuildSource(tmp_filename, None, None)],
            options=options,
        )

        mypy_file = result.files.get("__main__")
        if not mypy_file:
            os.unlink(tmp_filename)
            return
        self._visit_mypy_file(mypy_file)
        os.unlink(tmp_filename)

    def _visit_mypy_file(self, mypy_file: MypyFile) -> None:
        """Visit all top-level statements in the module."""
        self.module_symbol_table = mypy_file.names
        for stmt in mypy_file.defs:
            self._visit_stmt(stmt, scope=[self.module_name])

    def _visit_stmt(self, stmt: Statement, scope: list[str]) -> None:
        """Visit a statement and extract/check units as appropriate."""
        if isinstance(stmt, AssignmentStmt):
            for lvalue in stmt.lvalues:
                if isinstance(lvalue, NameExpr):
                    var_name = lvalue.name
                    key = ".".join([*scope, var_name])
                    unit = self._extract_unit_from_type(stmt)
                    if unit is not None:
                        self.units[key] = unit
                    # If assignment has a value, try to infer unit from the value
                    # (e.g., c = a + b)
                    if stmt.rvalue is not None:
                        inferred_unit = self._infer_unit_from_expr(stmt.rvalue, scope)
                        if isinstance(inferred_unit, Unit):
                            self.units[key] = inferred_unit
        elif isinstance(stmt, FuncDef):
            # Visit function body and check return units
            for substmt in stmt.body.body:
                self._visit_stmt(substmt, [*scope, stmt.name])

            # Record return type of function
            try:
                # class method
                sym_info = stmt.info.names.get(stmt.name)
                if not sym_info:
                    return
                sym_node = sym_info.node
                func_fullname = f"{stmt.info.fullname}.{stmt.name}"
            except AttributeError:
                # regular function
                sym_node = stmt
                func_fullname = ".".join([*scope, stmt.name])
            if not isinstance(sym_node, FuncDef):
                return
            unanalyzed_type = sym_node.unanalyzed_type
            if not isinstance(unanalyzed_type, CallableType) or not isinstance(
                unanalyzed_type.ret_type, UnboundType
            ):
                return
            ret_unit: Unit | TypeInfo
            if (ret_type := unanalyzed_type.ret_type).name == "Annotated":
                if not isinstance(ret_type.args[1], UnboundType):
                    return
                try:
                    ret_unit = Unit.from_string(ret_type.args[1].name)
                except IndexError:
                    return None
            else:
                if not isinstance(sym_node.type, CallableType) or not isinstance(
                    sym_node.type.ret_type, Instance
                ):
                    return None
                ret_unit = sym_node.type.ret_type.type
            self.function_returns[func_fullname] = ret_unit

            # Check all ReturnStmt nodes for unit correctness
            for substmt in stmt.body.body:
                if isinstance(substmt, ReturnStmt) and substmt.expr is not None:
                    returned_unit = self._infer_unit_from_expr(
                        substmt.expr, [*scope, stmt.name]
                    )
                    if returned_unit is not None and returned_unit != ret_unit:
                        self.errors.append(
                            UnitCheckerError(
                                code="U004",
                                lineno=getattr(substmt, "line", 0),
                                message=(
                                    "Unit of return value does not match function "
                                    f"signature: returned {returned_unit}, "
                                    f"expected {ret_unit}"
                                ),
                            )
                        )

        elif isinstance(stmt, ClassDef):
            class_name = getattr(stmt, "name", None)
            if class_name:
                new_scope = [*scope, class_name]
                for substmt in getattr(stmt.defs, "body", []):
                    self._visit_stmt(substmt, new_scope)
        elif isinstance(stmt, ExpressionStmt):
            self._infer_unit_from_expr(stmt.expr, scope)

    def _infer_unit_from_expr(
        self, expr: Expression, scope: list[str]
    ) -> TypeInfo | Unit | None:
        if isinstance(expr, NameExpr):
            if isinstance(expr.node, FuncDef):
                return self.function_returns.get(expr.fullname)
            elif isinstance(expr.node, TypeInfo):
                # breakpoint()
                return expr.node
            else:
                key = ".".join([*scope, expr.name])
                return self.units.get(key)
        elif isinstance(expr, UnaryExpr):
            # Handle unary operators: -a, +a, ~a
            operand_unit = self._infer_unit_from_expr(expr.expr, scope)
            # For unary ops, the unit is unchanged
            return operand_unit
        elif isinstance(expr, OpExpr):
            left = expr.left
            right = expr.right
            left_unit = self._infer_unit_from_expr(left, scope)
            right_unit = self._infer_unit_from_expr(right, scope)
            # if isinstance(left_unit, Unit) or isinstance(right_unit, Unit):
            #     return None
            op = expr.op
            if not isinstance(left_unit, Unit) or not isinstance(right_unit, Unit):
                self.errors.append(
                    UnitCheckerError(
                        code="U002",
                        lineno=getattr(expr, "line", 0),
                        message="Operands must both have units",
                    )
                )
                return None
            if op == "+" or op == "-":
                if left_unit == right_unit:
                    return left_unit
                else:
                    self.errors.append(
                        UnitCheckerError(
                            code="U001",
                            lineno=getattr(expr, "line", 0),
                            message=(
                                "Cannot add operands with different units: "
                                f"{left_unit} and {right_unit}"
                            ),
                        )
                    )
                    return None
            elif op == "*":
                return left_unit * right_unit
            elif op == "/":
                return left_unit * (right_unit**-1)
            else:
                return None
        elif isinstance(expr, CallExpr):
            # Try to resolve the object being called
            callee_unit_or_func = self._infer_unit_from_expr(expr.callee, scope)
            # Argument unit checking
            # If callee_unit_or_func is a FuncDef, check argument units
            if isinstance(expr.callee, NameExpr) and isinstance(
                expr.callee.node, FuncDef
            ):
                func_node = expr.callee.node
                # Get expected units for parameters
                param_units: list[Unit | None] = []
                unanalyzed_type = func_node.unanalyzed_type
                if not isinstance(unanalyzed_type, CallableType):
                    return None
                for arg_type in unanalyzed_type.arg_types:
                    if getattr(arg_type, "name", None) == "Annotated":
                        if not isinstance(arg_type, UnboundType) or not isinstance(
                            arg_type.args[1], UnboundType
                        ):
                            continue
                        try:
                            param_units.append(Unit.from_string(arg_type.args[1].name))
                        except ValueError:
                            param_units.append(None)
                    else:
                        param_units.append(None)
                # Check argument units
                for i, arg in enumerate(expr.args):
                    # Recursively check for nested CallExpr to catch mismatches in
                    # intermediate steps
                    if isinstance(arg, CallExpr):
                        self._infer_unit_from_expr(arg, scope)
                    arg_unit = self._infer_unit_from_expr(arg, scope)
                    expected_unit = param_units[i] if i < len(param_units) else None
                    if (
                        expected_unit is not None
                        and arg_unit is not None
                        and arg_unit != expected_unit
                    ):
                        self.errors.append(
                            UnitCheckerError(
                                code="U003",
                                lineno=getattr(expr, "line", 0),
                                message=(
                                    f"Argument {i + 1} to function '{func_node.name}' "
                                    f"has unit {arg_unit}, expected {expected_unit}"
                                ),
                            )
                        )
            return callee_unit_or_func
        elif isinstance(expr, MemberExpr):
            # Handle attribute access, e.g. f().a or obj.a
            base = expr.expr
            attr = expr.name
            # If base is a NameExpr (e.g., 'a' in 'a.a'), try to resolve its type
            if isinstance(base, NameExpr):
                base_name = base.name
                # Try to find the type of base_name in the symbol table
                if (
                    hasattr(self, "module_symbol_table")
                    and base_name in self.module_symbol_table
                ):
                    sym_node = self.module_symbol_table[base_name]
                    # If it's a variable with a type, and the type is an Instance
                    # then, get the class name
                    if isinstance(sym_node.node, Var) and hasattr(
                        sym_node.node, "type"
                    ):
                        var_type = sym_node.node.type
                        proper_type = get_proper_type(var_type)
                        if isinstance(proper_type, Instance):
                            class_fullname = proper_type.type.fullname
                            class_attr_key = f"{class_fullname}.{attr}"
                            if class_attr_key in self.units:
                                return self.units[class_attr_key]
                # Fallback: try previous logic for scope-based lookup
                for i in range(len(scope), 0, -1):
                    class_key = ".".join(scope[:i] + [base_name, attr])
                    if class_key in self.units:
                        return self.units[class_key]
            else:
                type_info = self._infer_unit_from_expr(base, scope)
                if not isinstance(type_info, TypeInfo):
                    return None
                if hasattr(type_info, "fullname"):
                    class_key = f"{type_info.fullname}.{attr}"
                    if class_key in self.function_returns:
                        return self.function_returns[class_key]
                    elif class_key in self.units:
                        return self.units[class_key]
            return None
        else:
            return None

    ANNOTATED_TYPE_NAMES = ("typing.Annotated", "typing_extensions.Annotated")

    def _extract_unit_from_type(self, stmt: AssignmentStmt) -> Unit | None:
        if (
            type_ := getattr(stmt, "unanalyzed_type", None)
        ) and type_.name == "Annotated":
            try:
                return Unit.from_string(type_.args[1].name)
            except IndexError:
                return None
        return None
