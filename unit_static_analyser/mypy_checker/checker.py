"""UnitChecker using mypy for type analysis.

This module provides a static analysis tool for extracting and checking physical units
from Python code using type annotations and mypy's typed AST.
"""

from graphlib import TopologicalSorter
from pathlib import Path

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
    IndexExpr,
    IntExpr,
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

analysis_result = Unit | TypeInfo | FuncDef | Instance | None


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
        self.function_units: dict[FuncDef, Unit | TypeInfo] = {}

    def _get_function_return_type(self, func_def: FuncDef) -> Unit | TypeInfo | None:
        unanalyzed_type = func_def.unanalyzed_type
        if not isinstance(unanalyzed_type, CallableType):
            return None
        ret_unit: Unit | TypeInfo | None = None
        if isinstance(unanalyzed_type.ret_type, UnboundType):
            if (ret_type := unanalyzed_type.ret_type).name == "Annotated":
                # return type has a unit annotation
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
                # no unit annotation but we may still need to capture the return type
                if not isinstance(func_def.type, CallableType) or not isinstance(
                    func_def.type.ret_type, Instance
                ):
                    return None
                ret_unit = func_def.type.ret_type.type
        return ret_unit

    def check(self, paths: list[Path]) -> None:
        """Run check units annotations on the given file(s) or directory(ies).

        Args:
            paths: List of file or directory paths to analyze.
        """
        files_and_modules = self._find_py_files_and_modules(paths)
        requested_modules = [module_name for _, module_name in files_and_modules]
        top_level_modules = self._get_top_level_modules(requested_modules)
        build_result = self._mypy_build(files_and_modules)
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
    def _mypy_build(files_and_modules: list[tuple[Path, str]]) -> BuildResult:
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

            inferred_unit = self._analyse_expression(stmt.rvalue)
            if isinstance(inferred_unit, Unit):
                self.var_units[key] = inferred_unit

    def _process_funcdef_stmt(self, stmt: FuncDef) -> None:
        """Process a function definition statement and extract/check units."""
        # store function arguments first so these are available when processing
        # the function body
        for argument in stmt.arguments:
            if (
                isinstance(argument.type_annotation, UnboundType)
                and argument.type_annotation.name == "Annotated"
            ):
                arg = argument.type_annotation.args[1]
                if not isinstance(arg, RawExpressionType) or not isinstance(
                    arg.literal_value, str
                ):
                    continue
                if arg.literal_value.startswith("unit:"):
                    self.var_units[argument.variable] = Unit.from_string(
                        arg.literal_value.removeprefix("unit:")
                    )

        for substmt in stmt.body.body:
            self._visit_stmt(substmt)

        return_type = self._get_function_return_type(stmt)
        if not return_type:
            return
        self.function_units[stmt] = return_type

        # Check all ReturnStmt nodes for unit correctness
        for substmt in stmt.body.body:
            if isinstance(substmt, ReturnStmt) and substmt.expr is not None:
                returned_unit = self._infer_unit_from_expression(substmt.expr)
                if isinstance(return_type, Unit) and returned_unit != return_type:
                    self.errors.append(
                        UnitCheckerError(
                            code="U004",
                            lineno=getattr(substmt, "line", 0),
                            message=(
                                "Unit of return value does not match function "
                                f"signature: returned {returned_unit}, "
                                f"expected {return_type}"
                            ),
                        )
                    )

    def _process_classdef_stmt(self, stmt: ClassDef) -> None:
        """Process a class definition statement and extract/check units."""
        for substmt in stmt.defs.body:
            self._visit_stmt(substmt)

    def _process_expression_stmt(self, stmt: ExpressionStmt) -> None:
        """Process an expression statement."""
        self._analyse_expression(stmt.expr)

    def _process_name_expr(self, expr: NameExpr) -> analysis_result:
        """Resolve a NameExpr to a Unit, TypeInfo, or FuncUnitDescription."""
        # Variable or symbol lookup
        if isinstance(expr.node, FuncDef | TypeInfo):
            return expr.node
        elif isinstance(expr.node, Var):
            if unit := self.var_units.get(expr.node):
                return unit
            elif isinstance(expr.node.type, Instance):
                return expr.node.type
            return None
        return None

    def _process_unary_expr(self, expr: UnaryExpr) -> analysis_result:
        """Resolve a UnaryExpr to its operand's unit (unary ops do not change units)."""
        return self._analyse_expression(expr.expr)

    def _process_power_op(
        self, expr: OpExpr, left_unit: analysis_result
    ) -> Unit | None:
        """Resolve a OpExpr raising value to a power."""
        if not isinstance(left_unit, Unit):
            return None
        if not isinstance(expr.right, IntExpr):
            self.errors.append(
                UnitCheckerError(
                    code="U009",
                    lineno=expr.line,
                    message="Exponent must be an explicit integer value.",
                )
            )
            return None
        exponent = expr.right.value
        return left_unit**exponent

    def _process_op_expr(self, expr: OpExpr) -> Unit | None:
        """Resolve an OpExpr (binary operator) to a unit, handling +, -, *, /."""
        left_unit = self._analyse_expression(expr.left)
        right_unit = self._analyse_expression(expr.right)
        op = expr.op
        has_left_unit = isinstance(left_unit, Unit)
        has_right_unit = isinstance(right_unit, Unit)

        if not has_left_unit and not has_right_unit:
            return None

        if op == "**":
            return self._process_power_op(expr, left_unit)

        if has_left_unit != has_right_unit:  # effective xor
            self.errors.append(
                UnitCheckerError(
                    code="U002",
                    lineno=getattr(expr, "line", 0),
                    message="Operands must both have units",
                )
            )
            return None

        if not isinstance(left_unit, Unit):
            return None
        if not isinstance(right_unit, Unit):
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

    def _infer_unit_from_expression(self, expr: Expression) -> Unit | None:
        analysed = self._analyse_expression(expr)
        if isinstance(analysed, Unit):
            return analysed
        return None

    def _analyse_expression(self, expr: Expression) -> analysis_result:
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
            case IndexExpr():
                return self._process_index_expr(expr)
            case _:
                return None

    def _check_arguments_of_function_call(
        self, func_def: FuncDef, args: list[Expression], line: int
    ) -> None:
        if func_def.info:
            func_arguments = func_def.arguments[1:]
        else:
            func_arguments = func_def.arguments

        for i, (arg, arg_def) in enumerate(zip(args, func_arguments)):
            expected_unit = self.var_units.get(arg_def.variable)
            if not expected_unit:
                continue
            inferred_unit = self._analyse_expression(arg)
            if inferred_unit != expected_unit:
                self.errors.append(
                    UnitCheckerError(
                        code="U003",
                        lineno=line,
                        message=(
                            f"Argument {i + 1} to function '{func_def.fullname}' "
                            f"has unit {inferred_unit}, expected {expected_unit}"
                        ),
                    )
                )

    def _process_call_expr(self, expr: CallExpr) -> analysis_result:
        """Resolve a CallExpr (function or constructor call) to a unit or type."""
        callee = self._analyse_expression(expr.callee)

        if isinstance(callee, FuncDef | Instance):
            if isinstance(callee, Instance):
                sym_node = callee.type.names.get("__call__")
                if not sym_node or not isinstance(sym_node.node, FuncDef):
                    return None
                func_def = sym_node.node
            else:
                func_def = callee
            self._check_arguments_of_function_call(func_def, expr.args, expr.line)
            return self.function_units.get(func_def)
        elif isinstance(callee, TypeInfo):
            # calling a class returns an instance
            return Instance(callee, [])
        return callee

    def _process_member_expr(
        self, expr: MemberExpr
    ) -> Unit | TypeInfo | FuncDef | None:
        """Resolve a MemberExpr (attribute access), including chained attributes."""
        if isinstance(expr.node, TypeInfo):
            return expr.node
        if isinstance(expr.node, Var) and expr.node in self.var_units:
            return self.var_units[expr.node]

        base = self._analyse_expression(expr.expr)

        if isinstance(base, TypeInfo | Instance):
            type_ = base if isinstance(base, TypeInfo) else base.type
            if sym_node := type_.names.get(expr.name):
                node = sym_node.node
                if isinstance(node, Var):
                    if isinstance(node.type, CallableType):
                        # accessing a class that is an attribute of another class
                        # lord knows why it's a CallableType
                        return node.type.type_object()
                    return self.var_units.get(node)
                elif isinstance(node, FuncDef):
                    return node
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
            left_unit = self._analyse_expression(operands[i])
            right_unit = self._analyse_expression(operands[i + 1])
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
        true_unit = self._analyse_expression(expr.if_expr)
        false_unit = self._analyse_expression(expr.else_expr)
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

    def _process_index_expr(self, expr: IndexExpr) -> analysis_result:
        base_unit = self._analyse_expression(expr.base)
        if isinstance(base_unit, Unit):
            return base_unit
        # callable type maps to __getitem__ method
        if isinstance(expr.method_type, CallableType):
            item_type = expr.method_type.ret_type
            if isinstance(item_type, CallableType):
                # don't love this think it needs a rework
                if isinstance(item_type.definition, FuncDef) and (
                    ret_type := self.function_units.get(item_type.definition)
                ):
                    return ret_type
                elif isinstance(item_type.ret_type, Instance):
                    return item_type.ret_type.type
            elif isinstance(expr.method_type.ret_type, Instance):
                return expr.method_type.ret_type
        return None
