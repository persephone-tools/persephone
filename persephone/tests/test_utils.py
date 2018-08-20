import pytest
def test_is_number():
    from persephone.utils import is_number

    assert is_number("12")
    assert is_number("12.0")
    assert not is_number("This isn't a number")


def test_make_batches():
    """Test that the batch generator creates the right batches"""
    from persephone.utils import make_batches
    paths = [1,2,3,4]

    batches1 = make_batches(paths, 1)
    assert len(batches1) == 4
    assert batches1 == [[1],[2],[3],[4]]

    batches2 = make_batches(paths, 2)
    assert len(batches2) == 2
    assert batches2 == [[1,2],[3,4]]

    batches10 = make_batches(paths, 10)
    assert len(batches10) == 1
    assert batches10 == [[1,2,3,4]]

def test_zero_batch_size():
    """Test that the batch generator raises exception if invalid parameter is passed"""
    from persephone.utils import make_batches
    paths = [1,2,3,4]

    with pytest.raises(ValueError):
        batches1 = make_batches(paths, 0)
