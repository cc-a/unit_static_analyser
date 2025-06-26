"""Types module."""

from typing import Annotated, TypeVar

from .core import Unit

T = TypeVar("T")
U = TypeVar("U", bound=Unit)

Quantity = Annotated[T, U]


# Example base units
class m(Unit):
    """Represents the meter unit."""

    pass


class s(Unit):
    """Represents the second unit."""

    pass


class kg(Unit):
    """Represents the kilogram unit."""

    pass
