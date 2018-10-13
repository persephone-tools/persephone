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

def test_model_train_and_decode(tmpdir, create_sine, make_wav, create_test_corpus):
    """Test that we can create a model, train it then decode something with it"""
    from persephone.corpus_reader import CorpusReader
    from persephone.rnn_ctc import Model
    from pathlib import Path
    corpus = create_test_corpus()

    # If it turns out that `tgt_dir` is not in the public interface of the Corpus
    # this test should change and get the base directory from the fixture that created it.
    base_directory = corpus.tgt_dir
    print("base_directory", base_directory)

    corpus_r = CorpusReader(
        corpus,
        batch_size=1
    )
    assert corpus_r

    test_model = Model(
        base_directory,
        corpus_r,
        num_layers=3,
        hidden_size=100
    )
    assert test_model

    test_model.train(
        early_stopping_steps=1,
        min_epochs=1,
        max_epochs=10
    )

    from persephone.model import decode

    wav_to_decode_path = str(tmpdir.join("to_decode.wav"))
    sine_to_decode = create_sine(note="C")

    make_wav(sine_to_decode, wav_to_decode_path)

    output_path = tmpdir.mkdir("decode_output")

    model_checkpoint_path = base_directory / "model" / "model_best.ckpt"
    result = decode(
        model_checkpoint_path,
        [Path(wav_to_decode_path)],
        label_set = {"A", "B", "C"},
        feature_type = "fbank",
        preprocessed_output_path=Path(str(output_path)),
        batch_x_name = test_model.batch_x.name,
        batch_x_lens_name = test_model.batch_x_lens.name,
        output_name = test_model.dense_decoded.name
    )

    assert result
    assert len(result) == 1


def test_model_train_callback(tmpdir, create_sine, make_wav, create_test_corpus):
    """Test that we can create a model, train it then get our callback called on each epoch of training"""
    from persephone.corpus_reader import CorpusReader
    from persephone.rnn_ctc import Model
    from pathlib import Path
    corpus = create_test_corpus()

    # If it turns out that `tgt_dir` is not in the public interface of the Corpus
    # this test should change and get the base directory from the fixture that created it.
    base_directory = corpus.tgt_dir
    print("base_directory", base_directory)

    corpus_r = CorpusReader(
        corpus,
        batch_size=1
    )
    assert corpus_r

    test_model = Model(
        base_directory,
        corpus_r,
        num_layers=3,
        hidden_size=50
    )
    assert test_model

    from unittest.mock import Mock

    mock_callback = Mock(return_value=None)

    test_model.train(
        early_stopping_steps=1,
        min_epochs=1,
        max_epochs=10,
        epoch_callback=mock_callback
    )

    assert mock_callback.call_count == 10
