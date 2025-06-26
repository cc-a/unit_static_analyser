"""Unit system: Unit class and base unit instances for use with typing.Annotated.

This module provides:
- The Unit class, representing physical units and supporting arithmetic.
- Base unit instances: m (meter), s (second), kg (kilogram).

Use these instances as metadata in typing.Annotated for static analysis and custom checkers.

Example:
    from typing import Annotated
    from .units import m, s

    distance: Annotated[int, m] = 100
    time: Annotated[int, s] = 20
"""

from typing import Any, Dict


class Unit:
    """Represents a physical unit as a mapping of base symbols to exponents."""

    def __init__(self, unit_map: Dict[str, int]):
        self.unit_map = {k: v for k, v in unit_map.items() if v != 0}

    def __mul__(self, other: "Unit") -> "Unit":
        symbols = set(self.unit_map) | set(other.unit_map)
        return Unit(
            {
                symbol: self.unit_map.get(symbol, 0) + other.unit_map.get(symbol, 0)
                for symbol in symbols
            }
        )

    def __truediv__(self, other: "Unit") -> "Unit":
        symbols = set(self.unit_map) | set(other.unit_map)
        return Unit(
            {
                symbol: self.unit_map.get(symbol, 0) - other.unit_map.get(symbol, 0)
                for symbol in symbols
            }
        )

    def __pow__(self, power: int) -> "Unit":
        return Unit({symbol: exp * power for symbol, exp in self.unit_map.items()})

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Unit):
            return False
        return self.unit_map == other.unit_map

    def __str__(self) -> str:
        num = []
        den = []
        for symbol, exp in self.unit_map.items():
            if exp > 0:
                num.append(f"{symbol}" + (f"^{exp}" if exp != 1 else ""))
            elif exp < 0:
                den.append(f"{symbol}" + (f"^{abs(exp)}" if exp != -1 else ""))
        num_str = "*".join(num) if num else "1"
        den_str = "*".join(den)
        return f"{num_str}/{den_str}" if den else num_str

    def __repr__(self) -> str:
        return f"Unit({self.unit_map})"


# Base unit instances
m = Unit({"m": 1})    # meter
s = Unit({"s": 1})    # second
kg = Unit({"kg": 1})  # kilogram
