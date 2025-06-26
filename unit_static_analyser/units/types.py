from typing import Annotated, TypeVar
from .core import Unit

T = TypeVar("T")
U = TypeVar("U", bound=Unit)

Quantity = Annotated[T, U]


# Example base units
class m(Unit):
    pass


class s(Unit):
    pass


class kg(Unit):
    pass
