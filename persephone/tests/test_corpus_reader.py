"""Tests for corpus reader functionality.

For convenience/brevity these tests use a Corpus object from a fixture."""

def test_corpus_reader(create_test_corpus):
    """Test that we can create a CorpusReader object"""
    from persephone.corpus_reader import CorpusReader
    corpus = create_test_corpus()
    corpus_r = CorpusReader(
        corpus,
        num_train=2,
        batch_size=1
    )
    assert corpus_r