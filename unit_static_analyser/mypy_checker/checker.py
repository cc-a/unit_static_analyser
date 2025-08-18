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
    Import,
    ImportFrom,
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
from mypy.types import CallableType, Instance, UnboundType

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
                if not isinstance(ret_type.args[1], UnboundType):
                    return None
                try:
                    ret_unit = Unit.from_string(ret_type.args[1].name)
                except IndexError:
                    return None
            else:
                if not isinstance(func_def.type, CallableType) or not isinstance(
                    func_def.type.ret_type, Instance
                ):
                    return None
                ret_unit = func_def.type.ret_type.type

        arg_units = dict()
        if isinstance(unanalyzed_type, CallableType):
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
                    if not isinstance(arg_type, UnboundType) or not isinstance(
                        arg_type.args[1], UnboundType
                    ):
                        continue
                    try:
                        # create FuncArgDescription
                        arg_units[arg_name] = FuncArgDescription(
                            position=unanalyzed_type.arg_names.index(arg_name) - offset,
                            name=arg_name,
                            unit=Unit.from_string(arg_type.args[1].name),
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
        result = []
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
        self.units: dict[str, Unit] = {}
        self.errors: list[UnitCheckerError] = []
        self.function_units: dict[str, FuncUnitDescription] = {}

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

        python_path_roots = set()
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
            self._visit_stmt(stmt, scope=[module_name])

    def _visit_stmt(self, stmt: Statement, scope: list[str]) -> None:
        """Visit a statement and extract/check units as appropriate."""
        match stmt:
            case AssignmentStmt():
                self._process_assignment_stmt(stmt, scope)
            case FuncDef():
                self._process_funcdef_stmt(stmt, scope)
            case ClassDef():
                self._process_classdef_stmt(stmt, scope)
            case ExpressionStmt():
                self._process_expression_stmt(stmt, scope)
            case ImportFrom():
                self._process_import_from_stmt(stmt, scope)
            case Import():
                self._process_import_stmt(stmt, scope)
            case _:
                pass

    def _process_import_from_stmt(self, stmt: ImportFrom, scope: list[str]) -> None:
        """Process an ImportFrom statement and update units/types for imported names."""
        module = stmt.id
        for name, as_name in stmt.names:
            imported_name = as_name or name
            key = f"{module}.{name}"
            # Try to find the unit/type in the source module
            if key in self.units:
                self.units[".".join([*scope, imported_name])] = self.units[key]
            elif key in self.function_units:
                self.function_units[".".join([*scope, imported_name])] = (
                    self.function_units[key]
                )

    def _process_import_stmt(self, stmt: Import, scope: list[str]) -> None:
        """Process an Import statement and update units/types for imported modules."""
        for module, as_name in stmt.ids:
            imported_name = as_name or module
            # Map the imported module name in the current scope to its full module name
            # This allows resolving e.g. b.x if 'import a as b'
            for key, value in list(self.units.items()):
                key_parts = key.split(".")
                if key_parts[0] == module:
                    self.units[".".join([*scope, imported_name, *key_parts[1:]])] = (
                        value
                    )
            # self.units[".".join([*scope, imported_name])] = module

    def _process_assignment_stmt(self, stmt: AssignmentStmt, scope: list[str]) -> None:
        """Process an assignment statement and extract/check units."""
        for lvalue in stmt.lvalues:
            if isinstance(lvalue, NameExpr):
                var_name = lvalue.name
                key = ".".join([*scope, var_name])
            elif (
                isinstance(lvalue, MemberExpr)
                and isinstance(lvalue.expr, NameExpr)
                and lvalue.expr.name == "self"
            ):
                # remove __init__ from the scope
                key = ".".join([*scope[:-1], lvalue.name])
            else:
                continue

            unit = self._extract_unit_from_type(stmt)
            if unit is not None:
                self.units[key] = unit
            # If assignment has a value, try to infer unit from the value
            # (e.g., c = a + b)
            if stmt.rvalue is not None:
                inferred_unit = self._infer_unit_from_expr(stmt.rvalue, scope)
                if isinstance(inferred_unit, Unit):
                    self.units[key] = inferred_unit
                if isinstance(inferred_unit, FuncUnitDescription):
                    if isinstance(inferred_unit.returns, Unit):
                        self.units[key] = inferred_unit.returns

    def _process_funcdef_stmt(self, stmt: FuncDef, scope: list[str]) -> None:
        """Process a function definition statement and extract/check units."""
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

        unit_desc = FuncUnitDescription.from_func_def(sym_node)
        if not unit_desc:
            return
        self.function_units[func_fullname] = unit_desc

        # populate function arguments into unit map
        for var_name, arg_desc in unit_desc.args.items():
            key = ".".join([*scope, stmt.name, var_name])
            if arg_desc.unit:
                self.units[key] = arg_desc.unit

        # Check all ReturnStmt nodes for unit correctness
        for substmt in stmt.body.body:
            if isinstance(substmt, ReturnStmt) and substmt.expr is not None:
                returned_unit = self._infer_unit_from_expr(
                    substmt.expr, [*scope, stmt.name]
                )
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

    def _lookup_unit_in_ascending_scopes(
        self, name: str, scope: list[str]
    ) -> Unit | None:
        for i in range(len(scope), 0, -1):
            key = ".".join([*scope[:i], name])
            if key in self.units:
                return self.units[key]
        return None

    def _lookup_function_in_ascending_scopes(
        self, name: str, scope: list[str]
    ) -> FuncUnitDescription | None:
        for i in range(len(scope), 0, -1):
            key = ".".join([*scope[:i], name])
            if key in self.function_units:
                return self.function_units[key]
        return None

    def _process_classdef_stmt(self, stmt: ClassDef, scope: list[str]) -> None:
        """Process a class definition statement and extract/check units."""
        class_name = getattr(stmt, "name", None)
        if class_name:
            new_scope = [*scope, class_name]
            for substmt in getattr(stmt.defs, "body", []):
                self._visit_stmt(substmt, new_scope)

    def _process_expression_stmt(self, stmt: ExpressionStmt, scope: list[str]) -> None:
        """Process an expression statement."""
        self._infer_unit_from_expr(stmt.expr, scope)

    def _process_name_expr(
        self, expr: NameExpr, scope: list[str]
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Resolve a NameExpr to a Unit, TypeInfo, or FuncUnitDescription."""
        # Variable or symbol lookup
        if isinstance(expr.node, FuncDef):
            return self._lookup_function_in_ascending_scopes(expr.name, scope)
        elif isinstance(expr.node, TypeInfo):
            return expr.node
        else:
            if unit := self._lookup_unit_in_ascending_scopes(expr.name, scope):
                return unit
            elif isinstance(expr.node, Var) and isinstance(expr.node.type, Instance):
                return expr.node.type
            return None

    def _process_unary_expr(
        self, expr: UnaryExpr, scope: list[str]
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Resolve a UnaryExpr to its operand's unit (unary ops do not change units)."""
        return self._infer_unit_from_expr(expr.expr, scope)

    def _process_op_expr(
        self, expr: OpExpr, scope: list[str]
    ) -> Unit | TypeInfo | FuncUnitDescription | None:
        """Resolve an OpExpr (binary operator) to a unit, handling +, -, *, /."""
        left_unit = self._infer_unit_from_expr(expr.left, scope)
        right_unit = self._infer_unit_from_expr(expr.right, scope)
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
        self, expr: Expression, scope: list[str]
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Recursively infer the unit for a given expression node.

        Args:
            expr: The mypy Expression node to analyze.
            scope: The current scope as a list of strings.

        Returns:
            The inferred Unit, TypeInfo, FuncUnitDescription, or None if not found.
        """
        match expr:
            case NameExpr():
                return self._process_name_expr(expr, scope)
            case UnaryExpr():
                return self._process_unary_expr(expr, scope)
            case OpExpr():
                return self._process_op_expr(expr, scope)
            case CallExpr():
                return self._process_call_expr(expr, scope)
            case MemberExpr():
                return self._process_member_expr(expr, scope)
            case ComparisonExpr():
                self._process_comparison_expr(expr, scope)
                return None
            case ConditionalExpr():
                return self._process_conditional_expr(expr, scope)
            case _:
                return None

    def _process_call_expr(
        self, expr: CallExpr, scope: list[str]
    ) -> Unit | TypeInfo | FuncUnitDescription | Instance | None:
        """Resolve a CallExpr (function or constructor call) to a unit or type."""
        callee = self._infer_unit_from_expr(expr.callee, scope)
        # Argument unit checking for functions
        if isinstance(callee, FuncUnitDescription) or isinstance(callee, Instance):
            func_desc = (
                callee
                if isinstance(callee, FuncUnitDescription)
                else self.function_units.get(f"{callee.type.fullname}.__call__")
            )
            if func_desc is None:
                return None
            for i, arg in enumerate(expr.args):
                arg_unit = self._infer_unit_from_expr(arg, scope)
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
        self, expr: MemberExpr, scope: list[str]
    ) -> Unit | TypeInfo | FuncUnitDescription | None:
        """Resolve a MemberExpr (attribute access), including chained attributes."""
        base = self._infer_unit_from_expr(expr.expr, scope)
        attr = expr.name

        if isinstance(base, FuncUnitDescription):
            if not base.returns:
                return None
            elif isinstance(base.returns, Unit):
                return base.returns
            else:
                # base is a TypeInfo
                class_key = f"{base.returns.fullname}.{attr}"
                if class_key in self.units:
                    # class attributes
                    return self.units[class_key]
                elif class_key in self.function_units:
                    # class methods
                    return self.function_units[class_key]
        elif isinstance(base, TypeInfo) or isinstance(base, Instance):
            type_ = base if isinstance(base, TypeInfo) else base.type

            if unit := self._lookup_unit_in_ascending_scopes(
                f"{type_.name}.{expr.name}", scope
            ):
                return unit
            elif func_desc := self._lookup_function_in_ascending_scopes(
                f"{type_.name}.{expr.name}", scope
            ):
                return func_desc

            if (sym_node := type_.get(attr)) is not None:
                if isinstance(sym_node.type, CallableType) and isinstance(
                    sym_node.type.ret_type, Instance
                ):
                    # accessing a class that is an attribute of another class
                    return sym_node.type.ret_type.type
                else:
                    raise NotImplementedError()
        return None

    ANNOTATED_TYPE_NAMES = ("typing.Annotated", "typing_extensions.Annotated")

    def _extract_unit_from_type(self, stmt: AssignmentStmt) -> Unit | None:
        """Extract a Unit from an assignment's type annotation if present.

        Args:
            stmt: The AssignmentStmt node to analyze.

        Returns:
            The extracted Unit if the type is Annotated, otherwise None.
        """
        if (
            type_ := getattr(stmt, "unanalyzed_type", None)
        ) and type_.name == "Annotated":
            try:
                return Unit.from_string(type_.args[1].name)
            except IndexError:
                return None

        return None

    def _process_comparison_expr(self, expr: ComparisonExpr, scope: list[str]) -> None:
        """Process a ComparisonExpr (e.g., a < b, x == y).

        Checks that all adjacent operands in the comparison have compatible units.
        Returns None, as comparisons are always unitless (boolean).
        """
        operands = expr.operands
        for i in range(len(operands) - 1):
            left_unit = self._infer_unit_from_expr(operands[i], scope)
            right_unit = self._infer_unit_from_expr(operands[i + 1], scope)
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

    def _process_conditional_expr(
        self, expr: ConditionalExpr, scope: list[str]
    ) -> Unit | None:
        """Process a ConditionalExpr (if-else expression).

        Returns the unit if both branches match, otherwise emits U007 and returns None.
        """
        true_unit = self._infer_unit_from_expr(expr.if_expr, scope)
        false_unit = self._infer_unit_from_expr(expr.else_expr, scope)
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
