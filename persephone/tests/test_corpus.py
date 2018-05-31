"""Tests for corpus related items"""
import pytest
def test_ready_corpus_deprecation():
    from persephone.corpus import ReadyCorpus
    with pytest.warns(DeprecationWarning):
        try:
            ReadyCorpus(tgt_dir="test_dir")
        except FileNotFoundError:
            pass


def test_determine_labels_throws():
    """Test that a non existant directory will throw"""
    import pathlib
    from persephone.corpus import determine_labels
    non_existent_path = "thispathdoesntexist"
    with pytest.raises(FileNotFoundError):
        determine_labels(non_existent_path, "phonemes")
