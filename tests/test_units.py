import pytest

from unit_static_analyser.units import Unit

def test_unit_from_string_simple():
    u = Unit.from_string("m")
    assert u.unit_map == {"m": 1}
    assert str(u) == "m"

def test_unit_from_string_power():
    u = Unit.from_string("s^-2")
    assert u.unit_map == {"s": -2}
    assert str(u) == "s^-2"

def test_unit_from_string_multiplication():
    u = Unit.from_string("kg.m^2.s^-2")
    assert u.unit_map == {"kg": 1, "m": 2, "s": -2}
    # Order in string may vary, so check all parts are present
    s = str(u)
    assert "kg" in s and "m^2" in s and "s^-2" in s

def test_unit_equality():
    u1 = Unit.from_string("kg.m^2.s^-2")
    u2 = Unit.from_string("kg.m^2.s^-2")
    u3 = Unit.from_string("kg.m^2")
    assert u1 == u2
    assert u1 != u3

def test_unit_multiplication():
    u1 = Unit.from_string("m")
    u2 = Unit.from_string("s^-2")
    result = u1 * u2
    assert result == Unit.from_string("m.s^-2")

def test_unit_power():
    u = Unit.from_string("m.s^-1")
    result = u ** 2
    assert result == Unit.from_string("m^2.s^-2")

def test_unit_zero_exponent_removed():
    u = Unit.from_string("m.s^-1")
    result = u * Unit.from_string("s")
    # m.s^-1 * s = m
    assert result == Unit.from_string("m")
    assert result.unit_map == {"m": 1}

def test_unit_invalid_string():
    with pytest.raises(ValueError):
        Unit.from_string("m//s")

def test_unit_repr_and_str():
    u = Unit.from_string("kg.m^2.s^-2")
    s = str(u)
    r = repr(u)
    assert "kg" in s and "m^2" in s and "s^-2" in s
    assert r.startswith("Unit(")
