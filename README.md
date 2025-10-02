# Unit Static Analyser

> **Warning:**
> This project is highly experimental and contains breaking changes and incomplete
> features. Use at your own risk.

Unit Static Analyser is a Python tool that can be used to verify the treatment of units
and physical quantities in a Python code base. A simple example:

```python
from typing import Annotated

distance: Annotated[int, "unit:m"] = 100
time: Annotated[int, "unit:s"] = 20
speed = distance / time  # Computes units as m/s
distance + time # disallowed
```

Using `typing.Annotated` allows arbitrary metadata to be attached alongside type
information. Unit Static Analyser then tracks the units through the code base and
ensures the operations performed are valid. In addition to using `typing.Annotated` you
can use type aliases:

```python
from typing import Annotated, TypeAlias, TypeVar

T = TypeVar("T")
meters: TypeAlias = Annotated[T, "unit:m"]
kg: TypeAlias = Annotated[T, "unit:kg"]

a: meters[int] = 1
b: kg[float] = 1.0
```

Unit Static Analyser is flexible and compatible with libraries like NumPy:

```python
from typing import Annotated
import numpy as np

a: Annotated[np.ndarray, "unit:m"] = np.array([])
```

Because it is a static analysis tools it avoids the compatibility drawbacks of some
runtime libraries for handling units.

A simple [demo project] is available showing application of Unit Static Analyser to a simple molecular dynamics code.

[demo project]: https://github.com/cc-a/unit_analysis_demo

## Usage

Note that though we use `Annotated` in the examples below, type aliases can be used just
as easily.

### Declaring Units

Unit Static Analyser makes no assumptions about the units you want to use. All units are
represented as strings with the prefix `"unit:"`. Units may be composed of multiple
parts separated by `"."`. Unit parts may also be raised to powers using `"^"`. For
instance `"unit:m.s^-1"` can be used to represent meters per second. Use of `"/"` is not
supported. To be compatible units must have the same parts raised to the same powers.

### Assigning Units

Units are assigned to variables via type annotations e.g.:

```python
a: Annotated[int, "unit:m"] = 1
```

Once a variable has a unit all subsequent expressions involving it are checked
for correctness of unit operations and units are propagated in assignments:

```python
a: Annotated[int, "unit:m"] = 1
b = a  # b now also has a unit of "m"
```

When assigning a unit to a variable any expression that does not have a unit associated (or has a compatible unit) will be accepted:

```python
a: Annotated[int, "unit:m"] = 1
b: Annotated[float, "unit:m"] = 1 / 3
c = np.ndarray([]) # c has no unit
d: Annotated[np.ndarray, "unit:m"] = c # but d does
```

Once a variable has a unit assigned, it cannot be changed later:

```python
a: Annotated[int, "unit:m"] = 1
a: Annotated[int, "unit:s"] = 1  # will report an error
```

### Arithmetic Operations

These behave as expected:

```python
a: Annotated[int, "unit:m"] = 1
b: Annotated[int, "unit:m"] = 2
c: Annotated[int, "unit:s"] = 3
a + b  # ok
b + c  # incompatible
a * b  # ok, gives m^2
a / c  # ok, gives m.s^-1
a**2  #  ok, gives m^2
(a / c)**-3 # ok, gives m^3.s^-3
```

All operands must have a unit associated with them. In order to work with scalar
quantities these must have an explicit annotation showing they have no unit:

```python
a: Annotated[int, "unit:m"] = 1
a * 3  # reports error
scalar: Annotated[int, "unit:"] = 3
a * scalar  # ok, gives unit of m
```

### Functions

Function arguments and return types can be assigned units:

```python
def f(a: Annotated[int, "unit:m"]) -> Annotated[int, "unit:m"]:
    return a
a: Annotated[int, "unit:m"] = 1
b = f(a) # b has unit of m
```

Arguments passed to functions are checked for unit compatibility. Using the above
example:

```python
f(5)
```

would report an error as 5 does not have unit information.

### Classes

Both attributes and methods can be annotated be with units:

```python
class A:
    a: Annotated[int, "unit:m"] = 1
    def method(self, b: Annotated[int, "unit:s"]):
        self.b: Annotated[int, "unit:s"] = b
```

Units for both `__init__` and `__call__` methods are supported.

### Containers

For types that support indexing it is assumed that the contents of the container have
the same unit as the container. So:

```python
a: Annotated[list[int], "unit:m"] = [0, 1]
b = a[0]  # b inherits unit of m
```

## Caveats and Limitations

- Loss of automatic type inference. Type checkers will often infer the type of an
  expression e.g. `a = 1` as int, without the need to provide an explicit type hint.
  However in order to add unit information we have to explicitly provide the type in the
  annotation.
- As it is built on top of Mypy it may not be compatible with other type checkers. In
  particular Pyright does not like the use of type aliases for declaring units.
- Limited understanding of third party libraries. Consider for instance:

  ```python
  a: Annotated[np.ndarray, "unit:m"] = np.array([])
  b = a.copy()
  ```

  Static Unit Analyser will record `b` as having no unit though intuitively one might
  expect it to have the same unit as `a`. This is because Unit Static Analyser doesn't
  know anything about the methods of NumPy arrays and how to correctly infer units. You can add a unit for `b` e.g.:

  ```python
  a: Annotated[np.ndarray, "unit:m"] = np.array([])
  b: Annotated[np.ndarray, "unit:m"] = a.copy()
  ```

  however there is no way for the tool to check correctness of this assignment.
- Whilst it should still be compatible with libraries that use `typing.Annotated` in
  other ways (e.g. Pydantic) the convenience and ergonomics of those libraries may be
  affected.
- Currently only a subset of all possible Python expressions are supported. The
  following are not yet supported:
  - Comprehensions (list, dict, etc)
  - Lambdas
  - Context Managers
  - Loops
  - Exception handling
  - Generators
  - Augmented assignments, e.g. `+=`
  - global and nonlocal statements

## For developers

This is a Python application that uses [poetry](https://python-poetry.org) for packaging
and dependency management. It also provides [pre-commit](https://pre-commit.com/) hooks
for various linters and formatters and automated tests using
[pytest](https://pytest.org/) and [GitHub Actions](https://github.com/features/actions).
Pre-commit hooks are automatically kept updated with a dedicated GitHub Action.

To get started:

1. [Download and install Poetry](https://python-poetry.org/docs/#installation) following the instructions for your OS.
1. Clone this repository and make it your working directory
1. Set up the virtual environment:

   ```bash
   poetry install
   ```

1. Activate the virtual environment (alternatively, ensure any Python-related command is preceded by `poetry run`):

   ```bash
   poetry shell
   ```

1. Install the git hooks:

   ```bash
   pre-commit install
   ```

1. Run the main app:

   ```bash
   python -m unit_static_analyser
   ```
