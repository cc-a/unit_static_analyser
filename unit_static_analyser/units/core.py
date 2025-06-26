from typing import Type, Dict


class UnitMeta(type):
    def __mul__(cls, other):
        return CompoundUnit._from_units(cls, other)

    def __truediv__(cls, other):
        return CompoundUnit._from_units(cls, other, div=True)

    def __pow__(cls, power):
        return CompoundUnit({cls: power})

    def __str__(cls):
        return cls.__name__


class Unit(metaclass=UnitMeta):
    pass


class CompoundUnit(Unit):
    def __init__(self, unit_map: Dict[Type[Unit], int]):
        self.unit_map = {k: v for k, v in unit_map.items() if v != 0}

    @classmethod
    def _from_units(cls, left, right, div=False):
        def to_map(u):
            if hasattr(u, "_unit_map"):
                return dict(u._unit_map)
            return {u: 1}

        left_map = to_map(left)
        right_map = to_map(right)
        unit_map = left_map.copy()
        for k, v in right_map.items():
            unit_map[k] = unit_map.get(k, 0) + (-v if div else v)
        return CompoundUnitFactory(unit_map)

    def __str__(self):
        num, den = [], []
        for unit, exp in self.unit_map.items():
            if exp > 0:
                num.append(f"{unit.__name__}" + (f"^{exp}" if exp != 1 else ""))
            elif exp < 0:
                den.append(f"{unit.__name__}" + (f"^{abs(exp)}" if exp != -1 else ""))
        num_str = "*".join(num) if num else "1"
        den_str = "*".join(den)
        return f"{num_str}/{den_str}" if den else num_str


_unit_cache = {}

def CompoundUnitFactory(unit_map: Dict[Type[Unit], int]):
    unit_map_tuple = tuple(sorted(unit_map.items(), key=lambda item: str(item[0])))  # Sort by string representation
    if unit_map_tuple in _unit_cache:
        return _unit_cache[unit_map_tuple]

    class _CompoundUnit(CompoundUnit):
        _unit_map = unit_map

    _unit_cache[unit_map_tuple] = _CompoundUnit
    return _CompoundUnit
