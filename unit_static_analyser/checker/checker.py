"""UnitChecker using mypy for type analysis.

This module provides a static analysis tool for extracting and checking physical units
from Python code using type annotations and mypy's typed AST.
"""

from dataclasses import dataclass
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
    OperatorAssignmentStmt,
    OpExpr,
    ReturnStmt,
    Statement,
    StrExpr,
    SymbolNode,
    TupleExpr,
    TypeAlias,
    TypeInfo,
    UnaryExpr,
    Var,
)
from mypy.options import Options
from mypy.types import (
    CallableType,
    Instance,
    RawExpressionType,
    Type,
    TypeAliasType,
    UnboundType,
    get_proper_type,
)

from ..units import Unit
from . import errors


@dataclass
class UnitNode:
    """Dataclass associating a unit and mypy node."""

    unit: Unit | None
    node: SymbolNode | Instance | None


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
        self.errors: list[errors.UnitCheckerError] = []
        self.units: dict[SymbolNode, Unit] = {}
        self.aliases: dict[TypeAlias, Unit] = {}

    def _get_function_return_unit(self, func_def: FuncDef) -> Unit | None:
        unanalyzed_type = func_def.unanalyzed_type
        if not isinstance(unanalyzed_type, CallableType):
            return None
        ret_unit: Unit | None = None
        if isinstance(unanalyzed_type.ret_type, UnboundType):
            if unit := self._extract_unit_from_type(unanalyzed_type.ret_type):
                return unit
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
            case OperatorAssignmentStmt():
                self._process_operator_assignment_stmt(stmt)
            case _:
                pass

    def _process_operator_assignment_stmt(self, stmt: OperatorAssignmentStmt) -> None:
        """Process an operator assignment statement (e.g., a += b)."""
        left_unit_node = self._analyse_expression(stmt.lvalue)
        right_unit = self._infer_unit_from_expression(stmt.rvalue)
        if bool(left_unit_node.unit) != bool(right_unit):
            self.errors.append(errors.u002_error_factory(lineno=stmt.line))
        elif left_unit_node.unit and right_unit:
            if stmt.op not in ("+", "-"):
                self.errors.append(
                    errors.u012_error_factory(lineno=stmt.line, operator=stmt.op)
                )
            elif left_unit_node.unit != right_unit:
                self.errors.append(
                    errors.u001_error_factory(
                        lineno=stmt.line,
                        left_unit=left_unit_node.unit,
                        right_unit=right_unit,
                    )
                )
        return None

    def _process_assignment_stmt(self, stmt: AssignmentStmt) -> None:
        """Process an assignment statement and extract/check units."""
        if stmt.is_alias_def:
            lvalue = stmt.lvalues[0]
            if not isinstance(lvalue, NameExpr):
                return None
            alias = self.module_symbol_table[lvalue.name].node
            if not isinstance(alias, TypeAlias):
                return None
            if not isinstance(stmt.rvalue, IndexExpr) or not isinstance(
                stmt.rvalue.base, NameExpr
            ):
                return None
            if stmt.rvalue.base.fullname == "typing.Annotated" and isinstance(
                stmt.rvalue.index, TupleExpr
            ):
                try:
                    item = stmt.rvalue.index.items[1]
                except IndexError:
                    return None
                if isinstance(item, StrExpr) and item.value.startswith("unit:"):
                    self.aliases[alias] = Unit.from_string(
                        item.value.removeprefix("unit:")
                    )
        else:
            for lvalue in stmt.lvalues:
                key = self._analyse_expression(lvalue)

                inferred_unit = self._infer_unit_from_expression(stmt.rvalue)

                annotated_unit: Unit | None = None
                if isinstance(key.node, Var):
                    if stmt.unanalyzed_type and isinstance(
                        stmt.unanalyzed_type, UnboundType
                    ):
                        annotated_unit = self._extract_unit_from_type(
                            stmt.unanalyzed_type
                        )

                if key.unit and annotated_unit and key.unit != annotated_unit:
                    self.errors.append(errors.u011_error_factory(lineno=stmt.line))
                elif annotated_unit and isinstance(key.node, Var):
                    self.units[key.node] = annotated_unit

                expected_unit = annotated_unit or key.unit

                if isinstance(inferred_unit, Unit):
                    if expected_unit and inferred_unit != expected_unit:
                        self.errors.append(
                            errors.u010_error_factory(
                                lineno=stmt.line,
                                fullname=key.node.fullname
                                if isinstance(key.node, Var)
                                else "expression",
                                expected_unit=expected_unit,
                                inferred_unit=inferred_unit,
                            )
                        )
                    if isinstance(key.node, Var):
                        self.units[key.node] = inferred_unit

    def _process_funcdef_stmt(self, stmt: FuncDef) -> None:
        """Process a function definition statement and extract/check units."""
        # store function arguments first so these are available when processing
        # the function body
        for argument in stmt.arguments:
            if isinstance(argument.type_annotation, UnboundType):
                unit = self._extract_unit_from_type(argument.type_annotation)
                if unit:
                    self.units[argument.variable] = unit

        for substmt in stmt.body.body:
            self._visit_stmt(substmt)

        return_unit = self._get_function_return_unit(stmt)
        if return_unit:
            self.units[stmt] = return_unit

        # Check all ReturnStmt nodes for unit correctness
        for substmt in stmt.body.body:
            if isinstance(substmt, ReturnStmt) and substmt.expr is not None:
                returned_unit = self._infer_unit_from_expression(substmt.expr)
                if return_unit and returned_unit != return_unit:
                    self.errors.append(
                        errors.u004_error_factory(
                            lineno=substmt.line,
                            returned_unit=returned_unit,
                            return_unit=return_unit,
                        )
                    )

    def _process_classdef_stmt(self, stmt: ClassDef) -> None:
        """Process a class definition statement and extract/check units."""
        for substmt in stmt.defs.body:
            self._visit_stmt(substmt)

    def _process_expression_stmt(self, stmt: ExpressionStmt) -> None:
        """Process an expression statement."""
        self._analyse_expression(stmt.expr)

    def _process_name_expr(self, expr: NameExpr) -> UnitNode:
        """Resolve a NameExpr to a Unit, TypeInfo, or FuncUnitDescription."""
        # Variable or symbol lookup
        if isinstance(expr.node, FuncDef | TypeInfo):
            return UnitNode(None, expr.node)
        elif isinstance(expr.node, Var):
            return UnitNode(self.units.get(expr.node), expr.node)
        elif expr.name in self.module_symbol_table:
            # this seems to come up rarely that a name expr has no node
            # in which case try a lookup in the module symbol table
            sym_node = self.module_symbol_table.get(expr.name)
            if sym_node and sym_node.node:
                return UnitNode(self.units.get(sym_node.node), sym_node.node)
        return UnitNode(None, None)

    def _process_unary_expr(self, expr: UnaryExpr) -> UnitNode:
        """Resolve a UnaryExpr to its operand's unit (unary ops do not change units)."""
        return self._analyse_expression(expr.expr)

    def _process_power_op(self, expr: OpExpr, left_unit: UnitNode) -> Unit | None:
        """Resolve a OpExpr raising value to a power."""
        if left_unit.unit is None:
            return None
        if not isinstance(expr.right, IntExpr):
            self.errors.append(errors.u009_error_factory(lineno=expr.line))
            return None
        exponent = expr.right.value
        return left_unit.unit**exponent

    def _process_op_expr(self, expr: OpExpr) -> UnitNode:
        """Resolve an OpExpr (binary operator) to a unit, handling +, -, *, /."""
        left_unit = self._analyse_expression(expr.left)
        right_unit = self._analyse_expression(expr.right)
        op = expr.op
        has_left_unit = isinstance(left_unit.unit, Unit)
        has_right_unit = isinstance(right_unit.unit, Unit)

        if not has_left_unit and not has_right_unit:
            # assume left and right nodes are interchangeable
            return UnitNode(None, left_unit.node)

        if op == "**":
            return UnitNode(self._process_power_op(expr, left_unit), left_unit.node)

        if has_left_unit != has_right_unit:  # effective xor
            self.errors.append(errors.u002_error_factory(lineno=expr.line))
            return UnitNode(None, left_unit.node)

        if not isinstance(left_unit.unit, Unit):
            return UnitNode(None, left_unit.node)
        if not isinstance(right_unit.unit, Unit):
            return UnitNode(None, left_unit.node)

        if op in {"+", "-"}:
            if left_unit.unit == right_unit.unit:
                return left_unit
            self.errors.append(
                errors.u001_error_factory(
                    lineno=expr.line,
                    left_unit=left_unit.unit,
                    right_unit=right_unit.unit,
                )
            )
            return UnitNode(None, left_unit.node)
        elif op == "*":
            return UnitNode(left_unit.unit * right_unit.unit, left_unit.node)
        elif op == "/":
            return UnitNode(left_unit.unit / right_unit.unit, left_unit.node)
        return UnitNode(None, left_unit.node)

    def _infer_unit_from_expression(self, expr: Expression) -> Unit | None:
        analysed = self._analyse_expression(expr)
        return analysed.unit

    def _analyse_expression(self, expr: Expression) -> UnitNode:
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
                return UnitNode(None, None)
            case ConditionalExpr():
                return self._process_conditional_expr(expr)
            case IndexExpr():
                return self._process_index_expr(expr)
            case _:
                return UnitNode(None, None)

    def _check_arguments_of_function_call(
        self, func_def: FuncDef, args: list[Expression], line: int
    ) -> None:
        if func_def.info:
            func_arguments = func_def.arguments[1:]
        else:
            func_arguments = func_def.arguments

        for i, (arg, arg_def) in enumerate(zip(args, func_arguments)):
            expected_unit = self.units.get(arg_def.variable)
            if not expected_unit:
                continue
            inferred_unit = self._analyse_expression(arg).unit
            if inferred_unit != expected_unit:
                self.errors.append(
                    errors.u003_error_factory(
                        lineno=line,
                        arg_index=i + 1,
                        func_fullname=func_def.fullname,
                        inferred_unit=inferred_unit,
                        expected_unit=expected_unit,
                    )
                )

    def _process_call_expr(self, expr: CallExpr) -> UnitNode:
        """Resolve a CallExpr (function or constructor call) to a unit or type."""
        callee = self._analyse_expression(expr.callee)

        node: Type | SymbolNode | Instance | None
        if isinstance(callee.node, Var):
            node = callee.node.type
        else:
            node = callee.node

        if isinstance(node, FuncDef | Instance):
            if isinstance(node, Instance):
                sym_node = node.type.names.get("__call__")
                if not sym_node or not isinstance(sym_node.node, FuncDef):
                    return UnitNode(None, None)
                func_def = sym_node.node
            else:
                func_def = node
            self._check_arguments_of_function_call(func_def, expr.args, expr.line)
            unit = self.units.get(func_def)
            if isinstance(func_def.type, CallableType):
                ret_type = None
                if isinstance(func_def.type.ret_type, Instance):
                    ret_type = func_def.type.ret_type
                elif isinstance(func_def.type.ret_type, TypeAliasType):
                    # deal with type aliases in the return type
                    proper_type = get_proper_type(func_def.type.ret_type)
                    if isinstance(proper_type, Instance):
                        ret_type = proper_type
                return UnitNode(unit, ret_type)
        elif isinstance(node, TypeInfo):
            # if __init__ is defined check arguments
            sym_node = node.names.get("__init__")
            if sym_node and isinstance(sym_node.node, FuncDef):
                self._check_arguments_of_function_call(
                    sym_node.node, expr.args, expr.line
                )
            # calling a class returns an instance
            return UnitNode(None, Instance(node, []))
        return callee

    def _process_member_expr(self, expr: MemberExpr) -> UnitNode:
        """Resolve a MemberExpr (attribute access), including chained attributes."""
        if isinstance(expr.node, TypeInfo | Var):
            # shortcut if mypy can already tell us what it is
            return UnitNode(self.units.get(expr.node), expr.node)

        base = self._analyse_expression(expr.expr)

        base_node = base.node.type if isinstance(base.node, Var) else base.node

        if isinstance(base_node, TypeInfo | Instance):
            type_ = base_node if isinstance(base_node, TypeInfo) else base_node.type
            if sym_node := type_.names.get(expr.name):
                node = sym_node.node
                if isinstance(node, Var):
                    if isinstance(node.type, CallableType):
                        # accessing a class that is an attribute of another class
                        # lord knows why it's a CallableType
                        return UnitNode(None, node.type.type_object())
                    return UnitNode(self.units.get(node), node)
                elif isinstance(node, FuncDef):
                    return UnitNode(None, node)
        return UnitNode(None, base.node)

    ANNOTATED_TYPE_NAMES = ("typing.Annotated", "typing_extensions.Annotated")

    def _extract_unit_from_type(self, type_: UnboundType) -> Unit | None:
        """Extract a Unit from an assignment's type annotation if present.

        Args:
            type_: The type to analyse.

        Returns:
            The extracted Unit if available, otherwise None.
        """
        if not (sym_node := self.module_symbol_table.get(type_.name)):
            return None

        if isinstance(sym_node.node, TypeAlias):
            return self.aliases.get(sym_node.node)
        elif (
            isinstance(sym_node.node, Var)
            and sym_node.node.fullname == "typing.Annotated"
        ):
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
            left_unit = self._infer_unit_from_expression(operands[i])
            right_unit = self._infer_unit_from_expression(operands[i + 1])
            if not isinstance(left_unit, Unit) and not isinstance(right_unit, Unit):
                return None
            # Disallow if only one side has units
            if isinstance(left_unit, Unit) != isinstance(right_unit, Unit):
                self.errors.append(errors.u006_error_factory(lineno=expr.line))
                return None
            # Both sides have unit
            if left_unit != right_unit:
                self.errors.append(
                    errors.u005_error_factory(
                        lineno=expr.line, left_unit=left_unit, right_unit=right_unit
                    )
                )
        # Comparison expressions are always unitless (boolean)
        return None

    def _process_conditional_expr(self, expr: ConditionalExpr) -> UnitNode:
        """Process a ConditionalExpr (if-else expression).

        Returns the unit if both branches match, otherwise emits U007 and returns None.
        """
        true_unit = self._analyse_expression(expr.if_expr)
        false_unit = self._analyse_expression(expr.else_expr)
        if isinstance(true_unit.unit, Unit) and isinstance(false_unit.unit, Unit):
            if true_unit.unit == false_unit.unit:
                return true_unit
            else:
                self.errors.append(
                    errors.u007_error_factory(
                        lineno=expr.line,
                        true_unit=true_unit.unit,
                        false_unit=false_unit.unit,
                    )
                )
                return UnitNode(None, true_unit.node)
        # If only one branch has a unit, treat as error (unit mismatch)
        if isinstance(true_unit.unit, Unit) != isinstance(
            false_unit.unit, Unit
        ):  # ^ is xor
            self.errors.append(errors.u008_error_factory(lineno=expr.line))
            return UnitNode(None, true_unit.node)
        # If neither branch has a unit, return None (unitless)
        return UnitNode(None, true_unit.node)

    def _process_index_expr(self, expr: IndexExpr) -> UnitNode:
        base_unit = self._analyse_expression(expr.base)

        # callable type maps to __getitem__ method
        if isinstance(expr.method_type, CallableType):
            item_type = expr.method_type.ret_type
            node = None
            if isinstance(item_type, CallableType):
                # don't love this think it needs a rework
                if isinstance(item_type.definition, FuncDef) and (
                    unit := self.units.get(item_type.definition)
                ):
                    return UnitNode(unit, item_type.definition)
                elif isinstance(item_type.ret_type, Instance):
                    return UnitNode(base_unit.unit, item_type.ret_type.type)
            elif isinstance(expr.method_type.ret_type, Instance):
                node = expr.method_type.ret_type
            return UnitNode(base_unit.unit, node)
        return UnitNode(base_unit.unit, None)
