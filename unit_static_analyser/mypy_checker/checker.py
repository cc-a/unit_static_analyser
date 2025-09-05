"""UnitChecker using mypy for type analysis.

This module provides a static analysis tool for extracting and checking physical units
from Python code using type annotations and mypy's typed AST.
"""

from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Self

from mypy.build import BuildResult, BuildSource, State, build
from mypy.nodes import (
    AssignmentStmt,
    CallExpr,
    ClassDef,
    ComparisonExpr,
    ConditionalExpr,
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
from mypy.types import CallableType, Instance, RawExpressionType, UnboundType

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


@dataclass
class FuncArgDescription:
    """Description of the units associated with a function argument."""

    position: int
    name: str
    unit: Unit | None
    kwarg: bool = False


@dataclass
class FuncUnitDescription:
    """Description of the units associated with a function."""

    returns: Unit | TypeInfo | None
    args: dict[str, FuncArgDescription]
    name: str
    fullname: str

    def get_arg_by_position(self, position: int) -> FuncArgDescription | None:
        """Get the argument of a function by its position."""
        for arg in self.args.values():
            if arg.position == position:
                return arg
        return None

    @classmethod
    def from_func_def(cls, func_def: FuncDef) -> Self | None:
        """Extract unit information from a function definition."""
        unanalyzed_type = func_def.unanalyzed_type
        if not isinstance(unanalyzed_type, CallableType):
            return None

        ret_unit: Unit | TypeInfo | None = None
        if isinstance(unanalyzed_type.ret_type, UnboundType):
            if (ret_type := unanalyzed_type.ret_type).name == "Annotated":
                if (
                    not isinstance(ret_type.args[1], RawExpressionType)
                    or not isinstance(ret_type.args[1].literal_value, str)
                    or not ret_type.args[1].literal_value.startswith("unit:")
                ):
                    return None
                try:
                    ret_unit = Unit.from_string(
                        ret_type.args[1].literal_value.removeprefix("unit:")
                    )
                except IndexError:
                    return None
            else:
                if not isinstance(func_def.type, CallableType) or not isinstance(
                    func_def.type.ret_type, Instance
                ):
                    return None
                ret_unit = func_def.type.ret_type.type

        arg_units: dict[str, FuncArgDescription] = dict()
        offset = 0
        for arg_name, arg_type in zip(
            unanalyzed_type.arg_names, unanalyzed_type.arg_types
        ):
            if arg_name is None:
                raise NotImplementedError("Unsupported form of function arguments.")
            if arg_name == "self":
                offset = 1
                continue
            if getattr(arg_type, "name", None) == "Annotated":
                if (
                    not isinstance(arg_type, UnboundType)
                    or not isinstance(arg_type.args[1], RawExpressionType)
                    or not isinstance(arg_type.args[1].literal_value, str)
                    or not arg_type.args[1].literal_value.startswith("unit:")
                ):
                    continue
                try:
                    # create FuncArgDescription
                    arg_units[arg_name] = FuncArgDescription(
                        position=unanalyzed_type.arg_names.index(arg_name) - offset,
                        name=arg_name,
                        unit=Unit.from_string(
                            arg_type.args[1].literal_value.removeprefix("unit:")
                        ),
                    )
                except ValueError:
                    pass

        return cls(
            returns=ret_unit,
            args=arg_units,
            name=func_def.name,
            fullname=".".join(func_def.fullname.split(".")[1:]),
        )


class UnitChecker:
    """Uses mypy's typed AST to extract and check units."""

    @staticmethod
    def _find_py_files_and_modules(paths: list[Path]) -> list[tuple[Path, str]]:
        """Find all Python files with module names from a list of paths.

        Returns:
            List of tuples: (absolute file path, module name)
        """
        result: list[tuple[Path, str]] = []
        for input_path in paths:
            input_path = input_path.resolve()
            if input_path.is_file() and input_path.suffix == ".py":
                module_name = UnitChecker._module_name_from_path(input_path)
                result.append((input_path, module_name))
            elif input_path.is_dir():
                for file_path in input_path.rglob("*.py"):
                    file_path = file_path.resolve()
                    module_name = UnitChecker._module_name_from_path(file_path)
                    result.append((file_path, module_name))
        return result

    @staticmethod
    def _module_name_from_path(file_path: "Path") -> str:
        """Compute the module name for a Python file, including all parent packages.

        For __init__.py, returns the package name.
        """
        # Walk up as long as __init__.py exists, for full package hierarchy
        if file_path.name == "__init__.py":
            parts = []
            current = file_path.parent
        else:
            parts = [file_path.with_suffix("").name]
            current = file_path.parent
        while (current / "__init__.py").exists():
            parts.insert(0, current.name)
            current = current.parent
        return ".".join(parts)

    @staticmethod
    def topological_sort_modules(
        graph: dict[str, State], requested_modules: list[str]
    ) -> list[str]:
        """Return a list of module names sorted in dependency order."""
        ts: TopologicalSorter[str] = TopologicalSorter()
        for mod, state in graph.items():
            if mod not in requested_modules:
                continue
            deps = [dep for dep in state.dependencies if dep in graph]
            ts.add(mod, *deps)
        return list(ts.static_order())

    def __init__(self) -> None:
        """Initialise a new checker.

        Args:
            module_name: module of interest
        """
        self.errors: list[UnitCheckerError] = []
        self.var_units: dict[Var, Unit] = {}
        self.function_units: dict[FuncDef, FuncUnitDescription] = {}

    def check(self, paths: list[Path]) -> None:
        """Run check units annotations on the given file(s) or directory(ies).

        Args:
            paths: List of file or directory paths to analyze.
        """
        files_and_modules = self._find_py_files_and_modules(paths)
        requested_modules = [module_name for _, module_name in files_and_modules]
        top_level_modules = self._get_top_level_modules(requested_modules)
        build_result = self._mypy_build(files_and_modules, top_level_modules)
        module_order = self.topological_sort_modules(
            build_result.graph, requested_modules
        )

        for module_name in self._get_modules_for_unit_analysis(
            module_order, top_level_modules
        ):
            if module_name.split(".")[0] not in top_level_modules:
                continue
            self._visit_mypy_file(build_result.files[module_name], module_name)

    @staticmethod
    def _mypy_build(
        files_and_modules: list[tuple[Path, str]], top_level_modules: set[str]
    ) -> BuildResult:
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

        python_path_roots: set[Path] = set()
        for file_path, module_name in files_and_modules:
            python_path_roots.add(
                Path(*file_path.parts[: -len(module_name.split("."))])
            )
        options.mypy_path = [str(path) for path in python_path_roots]

        sources = [
            BuildSource(str(file), module_name, None)
            for file, module_name in files_and_modules
        ]
        return build(sources=sources, options=options)

    @staticmethod
    def _get_top_level_modules(module_names: list[str]) -> set[str]:
        return {module_name.split(".")[0] for module_name in module_names}

    @staticmethod
    def _get_modules_for_unit_analysis(
        module_order: list[str], top_level_modules: set[str]
    ) -> list[str]:
        return [
            module_name
            for module_name in module_order
            if module_name.split(".")[0] in top_level_modules
        ]

    def _visit_mypy_file(self, mypy_file: MypyFile, module_name: str) -> None:
        """Visit all top-level statements in the module."""
        self.module_symbol_table = mypy_file.names
        for stmt in mypy_file.defs:
            self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: Statement) -> None:
        """Visit a statement and extract/check units as appropriate."""
        match stmt:
            case AssignmentStmt():
                self._process_assignment_stmt(stmt)
            case FuncDef():
                self._process_funcdef_stmt(stmt)
            case ClassDef():
                self._process_classdef_stmt(stmt)
            case ExpressionStmt():
                self._process_expression_stmt(stmt)
            case _:
                pass

    def _process_assignment_stmt(self, stmt: AssignmentStmt) -> None:
        """Process an assignment statement and extract/check units."""
        for lvalue in stmt.lvalues:
            if not isinstance(lvalue, NameExpr | MemberExpr):
                continue
            key = lvalue.node
            if not isinstance(key, Var):
                continue
            unit = self._extract_unit_from_type(stmt)
            if unit is not None:
                self.var_units[key] = unit
            # If assignment has a value, try to infer unit from the value
            # (e.g., c = a + b)
            inferred_unit = self._infer_unit_from_expr(stmt.rvalue)
            if isinstance(inferred_unit, Unit):
                self.var_units[key] = inferred_unit
            if isinstance(inferred_unit, FuncUnitDescription):
                if isinstance(inferred_unit.returns, Unit):
                    self.var_units[key] = inferred_unit.returns

    def _process_funcdef_stmt(self, stmt: FuncDef) -> None:
        """Process a function definition statement and extract/check units."""
        # Visit function body and check return units
        for substmt in stmt.body.body:
            self._visit_stmt(substmt)

        unit_desc = FuncUnitDescription.from_func_def(stmt)
        if not unit_desc:
            return
        self.function_units[stmt] = unit_desc

        # Check all ReturnStmt nodes for unit correctness
        for substmt in stmt.body.body:
            if isinstance(substmt, ReturnStmt) and substmt.expr is not None:
                returned_unit = self._infer_unit_from_expr(substmt.expr)
                if (
                    isinstance(returned_unit, Unit)
                    and returned_unit != unit_desc.returns
                ):
                    self.errors.append(
                        UnitCheckerError(
                            code="U004",
                            lineno=getattr(substmt, "line", 0),
                            message=(
                                "Unit of return value does not match function "
                                f"signature: returned {returned_unit}, "
                                f"expected {unit_desc.returns}"
                            ),
                        )
                    )

    def _process_classdef_stmt(self, stmt: ClassDef) -> None:
        """Process a class definition statement and extract/check units."""
        for substmt in stmt.defs.body:
            self._visit_stmt(substmt)

    def _process_expression_stmt(self, stmt: ExpressionStmt) -> None:
        """Process an expression statement."""
        self._infer_unit_from_expr(stmt.expr)

    def _process_name_expr(
        self, expr: NameExpr
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Resolve a NameExpr to a Unit, TypeInfo, or FuncUnitDescription."""
        # Variable or symbol lookup
        if isinstance(expr.node, FuncDef):
            return self.function_units[expr.node]
        elif isinstance(expr.node, TypeInfo):
            return expr.node
        elif isinstance(expr.node, Var):
            if unit := self.var_units.get(expr.node):
                return unit
            elif isinstance(expr.node.type, Instance):
                return expr.node.type
            return None
        return None

    def _process_unary_expr(
        self, expr: UnaryExpr
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Resolve a UnaryExpr to its operand's unit (unary ops do not change units)."""
        return self._infer_unit_from_expr(expr.expr)

    def _process_op_expr(
        self, expr: OpExpr
    ) -> Unit | TypeInfo | FuncUnitDescription | None:
        """Resolve an OpExpr (binary operator) to a unit, handling +, -, *, /."""
        left_unit = self._infer_unit_from_expr(expr.left)
        right_unit = self._infer_unit_from_expr(expr.right)
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
        if op in {"+", "-"}:
            if left_unit == right_unit:
                return left_unit
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
        return None

    def _infer_unit_from_expr(
        self, expr: Expression
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Recursively infer the unit for a given expression node.

        Args:
            expr: The mypy Expression node to analyze.

        Returns:
            The inferred Unit, TypeInfo, FuncUnitDescription, or None if not found.
        """
        match expr:
            case NameExpr():
                return self._process_name_expr(expr)
            case UnaryExpr():
                return self._process_unary_expr(expr)
            case OpExpr():
                return self._process_op_expr(expr)
            case CallExpr():
                return self._process_call_expr(expr)
            case MemberExpr():
                return self._process_member_expr(expr)
            case ComparisonExpr():
                self._process_comparison_expr(expr)
                return None
            case ConditionalExpr():
                return self._process_conditional_expr(expr)
            case _:
                return None

    def _process_call_expr(
        self, expr: CallExpr
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Resolve a CallExpr (function or constructor call) to a unit or type."""
        callee = self._infer_unit_from_expr(expr.callee)
        # Argument unit checking for functions
        if isinstance(callee, FuncUnitDescription) or isinstance(callee, Instance):
            if isinstance(callee, FuncUnitDescription):
                func_desc = callee
            else:
                if not (sym_node := callee.type.names.get("__call__")):
                    return None
                if not isinstance(sym_node.node, FuncDef):
                    return None
                func_desc = self.function_units[sym_node.node]
            for i, arg in enumerate(expr.args):
                arg_unit = self._infer_unit_from_expr(arg)
                arg_description = func_desc.get_arg_by_position(i)
                if not arg_description:
                    continue
                expected_unit = arg_description.unit
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
                                f"Argument {i + 1} to function '{func_desc.fullname}' "
                                f"has unit {arg_unit}, expected {expected_unit}"
                            ),
                        )
                    )
            return func_desc
        elif isinstance(callee, TypeInfo):
            # calling a class returns an instance
            return Instance(callee, [])
        return callee

    def _process_member_expr(
        self, expr: MemberExpr
    ) -> Unit | TypeInfo | FuncUnitDescription | None:
        """Resolve a MemberExpr (attribute access), including chained attributes."""
        if isinstance(expr.node, TypeInfo):
            return expr.node
        if isinstance(expr.node, Var) and expr.node in self.var_units:
            return self.var_units[expr.node]

        base = self._infer_unit_from_expr(expr.expr)

        if isinstance(base, FuncUnitDescription):
            if not base.returns:
                return None
            elif isinstance(base.returns, Unit):
                return base.returns
            else:
                type_ = base.returns
        elif isinstance(base, TypeInfo):
            type_ = base
        elif isinstance(base, Instance):
            type_ = base.type
        else:
            return None
        if not (node := type_.names[expr.name].node):
            return None
        if isinstance(node, Var):
            if isinstance(node.type, CallableType) and isinstance(
                node.type.ret_type, Instance
            ):
                # accessing a class that is an attribute of another class
                # lord knows why it's a CallableType
                return node.type.ret_type.type
            return self.var_units.get(node)
        elif isinstance(node, FuncDef):
            return self.function_units.get(node)
        return None

    ANNOTATED_TYPE_NAMES = ("typing.Annotated", "typing_extensions.Annotated")

    def _extract_unit_from_type(self, stmt: AssignmentStmt) -> Unit | None:
        """Extract a Unit from an assignment's type annotation if present.

        Args:
            stmt: The AssignmentStmt node to analyze.

        Returns:
            The extracted Unit if the type is Annotated, otherwise None.
        """
        type_ = stmt.unanalyzed_type
        if not isinstance(type_, UnboundType):
            return None
        if type_.name == "Annotated":
            arg = type_.args[1]
            if not isinstance(arg, RawExpressionType) or not isinstance(
                arg.literal_value, str
            ):
                return None
            if arg.literal_value.startswith("unit:"):
                return Unit.from_string(arg.literal_value.removeprefix("unit:"))
        return None

    def _process_comparison_expr(self, expr: ComparisonExpr) -> None:
        """Process a ComparisonExpr (e.g., a < b, x == y).

        Checks that all adjacent operands in the comparison have compatible units.
        Returns None, as comparisons are always unitless (boolean).
        """
        operands = expr.operands
        for i in range(len(operands) - 1):
            left_unit = self._infer_unit_from_expr(operands[i])
            right_unit = self._infer_unit_from_expr(operands[i + 1])
            if not isinstance(left_unit, Unit) and not isinstance(right_unit, Unit):
                return None
            # Disallow if only one side has units
            if isinstance(left_unit, Unit) ^ isinstance(right_unit, Unit):  # ^ is xor
                self.errors.append(
                    UnitCheckerError(
                        code="U006",
                        lineno=getattr(expr, "line", 0),
                        message=(
                            "Cannot compare a unitful operand with a unitless operand"
                        ),
                    )
                )
                return None
            # Both sides have unit
            if left_unit != right_unit:
                self.errors.append(
                    UnitCheckerError(
                        code="U005",
                        lineno=getattr(expr, "line", 0),
                        message=(
                            f"Cannot compare operands with different units: "
                            f"{left_unit} and {right_unit}"
                        ),
                    )
                )
        # Comparison expressions are always unitless (boolean)
        return None

    def _process_conditional_expr(self, expr: ConditionalExpr) -> Unit | None:
        """Process a ConditionalExpr (if-else expression).

        Returns the unit if both branches match, otherwise emits U007 and returns None.
        """
        true_unit = self._infer_unit_from_expr(expr.if_expr)
        false_unit = self._infer_unit_from_expr(expr.else_expr)
        if isinstance(true_unit, Unit) and isinstance(false_unit, Unit):
            if true_unit == false_unit:
                return true_unit
            else:
                self.errors.append(
                    UnitCheckerError(
                        code="U007",
                        lineno=getattr(expr, "line", 0),
                        message=(
                            f"Conditional branches have different units: "
                            f"{true_unit} and {false_unit}"
                        ),
                    )
                )
                return None
        # If only one branch has a unit, treat as error (unit mismatch)
        if isinstance(true_unit, Unit) ^ isinstance(false_unit, Unit):  # ^ is xor
            self.errors.append(
                UnitCheckerError(
                    code="U008",
                    lineno=getattr(expr, "line", 0),
                    message=("Both branches of conditional must a unit."),
                )
            )
            return None
        # If neither branch has a unit, return None (unitless)
        return None
