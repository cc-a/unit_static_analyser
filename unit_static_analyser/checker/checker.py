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

    def __init__(self, module_name="__main__") -> None:
        """Initialize the UnitChecker with an empty unit mapping and error list."""
        self.units: dict[str, Unit] = {}
        self.errors: list[UnitCheckerError] = []
        self.scope_stack: list[str] = [module_name]
        self.instance_map: dict[str, str] = {}  # maps instance name to class name
        self.function_return_units: dict[
            str, Unit
        ] = {}  # maps qualified function name to return unit
        self.current_function: list[str] = []  # stack of current function names

    def get_var_name(self, node: ast.AST) -> str | None:
        """Recursively build the full attribute chain for nodes like A.B.c.d"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self.get_var_name(node.value)
            if value is not None:
                # If value is an instance, resolve to its class
                if value in self.instance_map:
                    class_name = self.instance_map[value]
                    return f"{class_name}.{node.attr}"
                return f"{value}.{node.attr}"
        return None

    def scoped_key(self, var_name: str) -> str:
        # Always prefix with the full scope stack (which starts with module name)
        return ".".join(self.scope_stack + [var_name])

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
        # Track instance assignments: a = A()
        if (
            isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            self.instance_map[node.targets[0].id] = node.value.func.id
            # Also handle assignment from function call: b = f()
            func_name = node.value.func.id
            # Try all possible qualified names from innermost to outermost
            for i in range(len(self.scope_stack), -1, -1):
                scope = self.scope_stack[:i]
                qualified_func_name = (
                    ".".join(scope + [func_name]) if scope else func_name
                )
                if qualified_func_name in self.function_return_units:
                    self.units[self.scoped_key(node.targets[0].id)] = (
                        self.function_return_units[qualified_func_name]
                    )
                    break
            return

        # Track variable aliasing: b = a
        if (
            isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.instance_map
        ):
            self.instance_map[node.targets[0].id] = self.instance_map[node.value.id]
            return

        # Assignment from function call: b = f()
        if (
            isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            func_name = node.value.func.id
            if func_name in self.function_return_units:
                self.units[self.scoped_key(node.targets[0].id)] = (
                    self.function_return_units[func_name]
                )
            return

        if isinstance(node.value, ast.BinOp):
            left_var = node.value.left
            right_var = node.value.right
            op = node.value.op

            if not (isinstance(node.targets[0], ast.Name)):
                return

            target = node.targets[0].id

            left_var_name = self.get_var_name(left_var)
            right_var_name = self.get_var_name(right_var)

            left_unit = self.lookup_unit(left_var_name) if left_var_name else None
            right_unit = self.lookup_unit(right_var_name) if right_var_name else None

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
                result_unit = left_unit * (right_unit**-1)
                self.units[self.scoped_key(target)] = result_unit
            elif isinstance(op, ast.Pow):
                if isinstance(node.value.right, ast.Constant) and isinstance(
                    node.value.right.value, int
                ):
                    result_unit = left_unit**node.value.right.value
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
        # Track function return unit if annotated
        return_unit = (
            self._extract_unit_from_annotation(node.returns) if node.returns else None
        )
        qualified_func_name = ".".join(self.scope_stack)
        if return_unit is not None:
            self.function_return_units[qualified_func_name] = Unit.from_string(
                return_unit
            )
        self.current_function.append(node.name)
        self.generic_visit(node)
        self.current_function.pop()
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_stack.append(node.name)
        # Handle inheritance: copy units from base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_class = base.id
                # Copy all units from base_class.<var> to current_class.<var>
                # Use fully qualified names for prefix and current_prefix
                prefix = ".".join(self.scope_stack[:-1] + [base_class]) + "."
                current_prefix = ".".join(self.scope_stack) + "."
                for key, unit in list(self.units.items()):
                    if key.startswith(prefix):
                        new_key = current_prefix + key[len(prefix) :]
                        self.units[new_key] = unit
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Return(self, node: ast.Return) -> None:
        # Only check if inside a function with a return annotation
        if not self.current_function:
            return
        func_name = self.current_function[-1]
        qualified_func_name = ".".join(self.scope_stack)
        expected_unit = self.function_return_units.get(qualified_func_name)
        if expected_unit is None:
            # If the returned value has a unit, but the function signature does not, error
            return_var_name = self.get_var_name(node.value) if node.value else None
            actual_unit = self.lookup_unit(return_var_name) if return_var_name else None
            if actual_unit is not None:
                self.errors.append(
                    UnitCheckerError(
                        code="U005",
                        lineno=node.lineno,
                        message="Units of returned value does not match function signature: returned={}, expected={}".format(
                            actual_unit, expected_unit
                        ),
                    )
                )
            return
        # Try to get the unit of the returned value
        return_var_name = self.get_var_name(node.value) if node.value else None
        actual_unit = self.lookup_unit(return_var_name) if return_var_name else None
        if actual_unit is None:
            # Try to handle constant returns (e.g., return 1)
            if isinstance(node.value, ast.Constant):
                actual_unit = None
        if actual_unit != expected_unit:
            self.errors.append(
                UnitCheckerError(
                    code="U005",
                    lineno=node.lineno,
                    message="Units of returned value does not match function signature: returned={}, expected={}".format(
                        actual_unit, expected_unit
                    ),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Simulate: from other_module import var_with_units
        module = node.module
        for alias in node.names:
            imported_name = alias.name
            asname = alias.asname or imported_name
            imported_key = f"{module}.{imported_name}"
            local_key = self.scoped_key(asname)
            if imported_key in self.units:
                self.units[local_key] = self.units[imported_key]

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
                    if isinstance(unit_elt, ast.Constant) and isinstance(
                        unit_elt.value, str
                    ):
                        return unit_elt.value  # Extract string literal
        return None
