import pytest

def test_na():
    import persephone.datasets.na

def test_butcher():
    import persephone.datasets.butcher

@pytest.mark.skip(reason="Currently this is not functional. Needs fixes before it van be imported.")
def test_kunwinjku_steven():
    import persephone.datasets.kunwinjku_steven

@pytest.mark.skip(reason="Currently this is not functional. Needs fixes before it van be imported.")
def test_chatino():
    import persephone.datasets.chatino

def test_pangloss():
    import persephone.datasets.pangloss
