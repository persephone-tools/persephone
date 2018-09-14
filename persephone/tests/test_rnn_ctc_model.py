"""Test that we can create an RNN CTC model"""

def test_model_creation(create_test_corpus):
    """Test that we can create a model"""
    from persephone.corpus_reader import CorpusReader
    from persephone.rnn_ctc import Model
    corpus = create_test_corpus()
    corpus_r = CorpusReader(
        corpus,
        num_train=1,
        batch_size=1
    )
    assert corpus_r

    model = Model(
        corpus.tgt_dir,
        corpus_r,
    )
    assert model