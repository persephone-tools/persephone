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

def test_model_train_and_decode(tmpdir, create_test_corpus):
    """Test that we can create a model, train it then decode something with it"""
    from persephone.corpus_reader import CorpusReader
    from persephone.rnn_ctc import Model
    corpus = create_test_corpus()

    # If it turns out that `tgt_dir` is not in the public interface this test should change
    # to get the base directory from the fixture that created it.
    base_directory = corpus.tgt_dir
    print("base_directory", base_directory)

    corpus_r = CorpusReader(
        corpus,
        num_train=1,
        batch_size=1
    )
    assert corpus_r

    test_model = Model(
        base_directory,
        corpus_r,
        num_layers=2,
        hidden_size=10
    )
    assert test_model

    test_model.train(
        early_stopping_steps=1,
        min_epochs=1,
        max_epochs=10
    )

    from persephone.model import decode

    wav_to_decode_path = Path(tmpdir.join("to_decide.wav"))
    sine_to_decode = create_sine(note="C")

    make_wav(sine_to_decode, wav_to_decode_path)

    result = decode(
        base_directory,
        [wav_to_decode_path],
        label_set = {"a", "b", "c"},
        feature_type = "fbank",
        batch_x_name = test_model.batch_x,
        batch_x_lens_name = test_model.batch_x_lens,
        output_name = test_model.outputs
    )

    assert result
    assert len(result) == 1