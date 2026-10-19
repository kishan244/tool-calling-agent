from tools import calculator


def test_basic_arithmetic():
    assert calculator("847 * 293") == "248171"
    assert calculator("(2 + 3) * 4") == "20"
    assert calculator("-5 + 2") == "-3"


def test_division_by_zero_is_handled():
    assert "division by zero" in calculator("1 / 0")


def test_code_is_rejected():
    assert calculator("__import__('os').system('ls')").startswith("Calculation failed")


def test_huge_power_is_blocked():
    assert "too large" in calculator("9 ** 9 ** 9")
