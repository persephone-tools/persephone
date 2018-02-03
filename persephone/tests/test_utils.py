def test_is_number():
    from persephone.utils import is_number

    assert is_number("12")
    assert is_number("12.0")
    assert not is_number("This isn't a number")
