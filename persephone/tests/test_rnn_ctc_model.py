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

def test_model_train_and_decode(tmpdir, create_sine, create_note_sequence, make_wav, create_test_corpus):
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
        min_epochs=3,
        max_epochs=10
    )

    from persephone.model import decode

    wav_to_decode_path1 = str(tmpdir.join("to_decode1.wav"))
    notes_to_decode1 = ["A", "C"]
    data_to_decode1 = create_note_sequence(notes=notes_to_decode1)
    make_wav(data_to_decode1, wav_to_decode_path1)

    wav_to_decode_path2 = str(tmpdir.join("to_decode2.wav"))
    notes_to_decode2 = ["C", "B"]
    data_to_decode2 = create_note_sequence(notes=notes_to_decode2)
    make_wav(data_to_decode2, wav_to_decode_path2)

    output_path = tmpdir.mkdir("decode_output")

    model_checkpoint_path = base_directory / "model" / "model_best.ckpt"
    result = decode(
        model_checkpoint_path,
        [Path(wav_to_decode_path1), Path(wav_to_decode_path2)],
        label_set = {"A", "B", "C"},
        feature_type = "fbank",
        preprocessed_output_path=Path(str(output_path)),
        batch_x_name = test_model.batch_x.name,
        batch_x_lens_name = test_model.batch_x_lens.name,
        output_name = test_model.dense_decoded.name
    )

    # Make sure the model hypothesis is correct
    assert result == [notes_to_decode1, notes_to_decode2]

    assert result
    assert len(result) == 1
