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

import re
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
        if not self.unit_map:
            return "1"
        parts = []
        for symbol in sorted(self.unit_map):  # sort for consistency
            exp = self.unit_map[symbol]
            if exp == 1:
                parts.append(f"{symbol}")
            else:
                parts.append(f"{symbol}^{exp}")
        return ".".join(parts)

    def __repr__(self) -> str:
        return f"Unit({self.unit_map})"

    @classmethod
    def from_string(cls, unit_str: str) -> "Unit":
        """
        Parse a string like 'kg.m^2.s^-2' into a Unit instance.

        - Multiplication: '.'
        - Powers: '^'
        - No division allowed.
        """

        unit_map = {}
        for part in unit_str.split("."):
            match = re.fullmatch(r"([a-zA-Z]+)(?:\^(-?\d+))?", part)
            if not match:
                raise ValueError(f"Invalid unit part: {part}")
            symbol = match.group(1)
            exp = int(match.group(2)) if match.group(2) else 1
            unit_map[symbol] = unit_map.get(symbol, 0) + exp
        if not unit_map:
            raise ValueError(f"Invalid unit string: {unit_str}")
        return cls(unit_map)
