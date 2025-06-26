"""Metaclass infrastructure for unit types."""


class UnitMeta(type):
    """Metaclass for defining unit operations like multiplication, division, etc."""

    def __mul__(cls: type, other: type) -> type:
        """Multiply two units to create a compound unit."""
        return CompoundUnit._from_units(cls, other)

    def __truediv__(cls: type, other: type) -> type:
        """Divide two units to create a compound unit."""
        return CompoundUnit._from_units(cls, other, div=True)

    def __pow__(cls: type, power: int) -> type:
        """Raise a unit to a power to create a compound unit."""
        return CompoundUnitFactory({cls: power})

    def __str__(cls) -> str:
        """Return the string representation of the unit."""
        return cls.__name__


class Unit(metaclass=UnitMeta):
    """Base class for all units."""

    pass


class CompoundUnit(Unit):
    """Represents a compound unit formed by combining base units."""

    def __init__(self, unit_map: dict[type["Unit"], int]) -> None:
        """Initialize a compound unit with a mapping of base units to their powers."""
        self.unit_map: dict[type, int] = {k: v for k, v in unit_map.items() if v != 0}

    @classmethod
    def _from_units(cls, left: type, right: type, div: bool = False) -> type:
        """Create a compound unit from two units, optionally dividing them."""

        def to_map(u: type) -> dict[type, int]:
            """Convert a unit to its unit map representation."""
            if hasattr(u, "_unit_map"):
                return dict(u._unit_map)
            return {u: 1}

        left_map = to_map(left)
        right_map = to_map(right)
        unit_map = left_map.copy()
        for k, v in right_map.items():
            unit_map[k] = unit_map.get(k, 0) + (-v if div else v)
        return CompoundUnitFactory(unit_map)

    def __str__(self) -> str:
        """Return the string representation of the compound unit."""
        num, den = [], []
        for unit, exp in self.unit_map.items():
            if exp > 0:
                num.append(f"{unit.__name__}" + (f"^{exp}" if exp != 1 else ""))
            elif exp < 0:
                den.append(f"{unit.__name__}" + (f"^{abs(exp)}" if exp != -1 else ""))
        num_str = "*".join(num) if num else "1"
        den_str = "*".join(den)
        return f"{num_str}/{den_str}" if den else num_str


_unit_cache: dict[tuple[tuple[type, int], ...], type] = {}


def CompoundUnitFactory(unit_map: dict[type, int]) -> type:
    """Factory function to create or retrieve cached compound units."""
    unit_map_tuple = tuple(
        sorted(unit_map.items(), key=lambda item: str(item[0]))
    )  # Sort by string representation
    if unit_map_tuple in _unit_cache:
        return _unit_cache[unit_map_tuple]

    class _CompoundUnit(CompoundUnit):
        """Internal class representing a compound unit."""

        _unit_map = unit_map

    _unit_cache[unit_map_tuple] = _CompoundUnit
    return _CompoundUnit
