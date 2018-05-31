"""Tests for corpus related items"""
import pytest
def test_ready_corpus_deprecation():
    from persephone.corpus import ReadyCorpus
    with pytest.warns(DeprecationWarning):
        try:
            ReadyCorpus(tgt_dir="test_dir")
        except FileNotFoundError:
            pass