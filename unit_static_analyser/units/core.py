"""Metaclass infrastructure for unit types for static analysis.

This module enables type-level unit arithmetic (e.g., m / s) so that units can be used
as type parameters (e.g., Quantity[int, m]) for static analysis and type checking.
"""


class UnitMeta(type):
    """Metaclass for unit types, enabling type-level arithmetic."""

    def __mul__(cls: type, other: type) -> type:
        """Multiply two unit types to create a compound unit type."""
        return _compound_unit_factory(
            _combine_unit_maps(_unit_map(cls), _unit_map(other), add=True)
        )

    def __truediv__(cls: type, other: type) -> type:
        """Divide two unit types to create a compound unit type."""
        return _compound_unit_factory(
            _combine_unit_maps(_unit_map(cls), _unit_map(other), add=False)
        )

    def __pow__(cls: type, power: int) -> type:
        """Raise a unit type to a power to create a compound unit type."""
        return _compound_unit_factory({cls: power})

    def __str__(cls) -> str:
        """Return the string representation of the unit type."""
        return cls.__name__


class Unit(metaclass=UnitMeta):
    """Base class for all unit types."""

    pass


def _unit_map(unit: type) -> dict[type, int]:
    """Return the unit map for a unit type."""
    return getattr(unit, "_unit_map", {unit: 1})


def _combine_unit_maps(
    left: dict[type, int], right: dict[type, int], add: bool = True
) -> dict[type, int]:
    """Combine two unit maps for multiplication or division."""
    result = left.copy()
    for k, v in right.items():
        result[k] = result.get(k, 0) + (v if add else -v)
    # Remove zero exponents
    return {k: v for k, v in result.items() if v != 0}


_unit_cache: dict[frozenset, type] = {}


def _compound_unit_factory(unit_map: dict[type, int]) -> type:
    """Create or retrieve a compound unit type from a unit map."""
    key = frozenset(unit_map.items())
    if key in _unit_cache:
        return _unit_cache[key]

    class CompoundUnit(Unit):
        _unit_map = unit_map

        def __str__(self) -> str:
            num, den = [], []
            for unit, exp in self._unit_map.items():
                if exp > 0:
                    num.append(f"{unit.__name__}" + (f"^{exp}" if exp != 1 else ""))
                elif exp < 0:
                    den.append(
                        f"{unit.__name__}" + (f"^{abs(exp)}" if exp != -1 else "")
                    )
            num_str = "*".join(num) if num else "1"
            den_str = "*".join(den)
            return f"{num_str}/{den_str}" if den else num_str

    _unit_cache[key] = CompoundUnit
    return CompoundUnit
