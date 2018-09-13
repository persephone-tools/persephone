"""Tests for corpus related items"""
import pytest

def test_corpus_import():
    """Test we can import Corpus"""
    from persephone.corpus import Corpus

def test_missing_experiment_dir():
    """A Corpus needs an experiment directory, check an exception is thrown
    if the directory doesn't exist"""
    from pathlib import Path
    from persephone.corpus import Corpus

    with pytest.raises(FileNotFoundError):
        Corpus(
            feat_type='fbank',
            label_type='phonemes',
            tgt_dir=Path("thisDoesNotExist"),
            labels=["a", "b", "c"]
        )

def test_missing_wav_dir(tmpdir):
    """Test that a missing wav dir raises an error"""
    from pathlib import Path
    from persephone.corpus import Corpus
    from persephone.exceptions import PersephoneException

    with pytest.raises(PersephoneException):
        Corpus(
            feat_type='fbank',
            label_type='phonemes',
            tgt_dir=Path(str(tmpdir)),
            labels=["a", "b", "c"]
        )

def test_create_corpus_no_data(tmpdir):
    """Test that an attempt to create a Corpus object with no data raises an
    exception warning us that there's no data"""
    from persephone.corpus import Corpus
    from pathlib import Path

    wav_dir = tmpdir.mkdir("wav")
    label_dir = tmpdir.mkdir("label")

    from persephone.exceptions import PersephoneException

    with pytest.raises(PersephoneException):
        c = Corpus(
                feat_type='fbank',
                label_type='phonemes',
                tgt_dir=Path(str(tmpdir)),
                labels=["a", "b", "c"]
            )


def test_create_corpus_basic(tmpdir, create_sine, make_wav):
    """Test that an attempt to create a Corpus object with a minimal data set"""
    from persephone.corpus import Corpus
    from pathlib import Path

    wav_dir = tmpdir.mkdir("wav")
    label_dir = tmpdir.mkdir("label")

    #create sine wave data
    data_a = create_sine(note="A")
    data_b = create_sine(note="B")
    data_c = create_sine(note="C")

    wav_test = wav_dir.join("test.wav")
    make_wav(data_a, str(wav_test))
    wav_train = wav_dir.join("train.wav")
    make_wav(data_b, str(wav_train))
    wav_valid = wav_dir.join("valid.wav")
    make_wav(data_c, str(wav_valid))

    label_test = label_dir.join("test.phonemes").write("a")
    label_train = label_dir.join("train.phonemes").write("b")
    label_valid = label_dir.join("valid.phonemes").write("c")

    c = Corpus(
        feat_type='fbank',
        label_type='phonemes',
        tgt_dir=Path(str(tmpdir)),
        labels=None
    )
    assert c


def test_corpus_with_predefined_data_sets(tmpdir, create_sine, make_wav):
    """Test that corpus construction works with prefix data splits determined
    as per the file system conventions.

    This will check that what is specified in :
    * `test_prefixes.txt`
    * `train_prefixes.txt`
    * `valid_prefixes.txt`
    Matches the internal members that store the prefix information
    """
    from persephone.corpus import Corpus
    from pathlib import Path

    wav_dir = tmpdir.mkdir("wav")
    label_dir = tmpdir.mkdir("label")

    #create sine wave data
    data_a = create_sine(note="A")
    data_b = create_sine(note="B")
    data_c = create_sine(note="C")

    wav_test = wav_dir.join("test.wav")
    make_wav(data_a, str(wav_test))
    wav_train = wav_dir.join("train.wav")
    make_wav(data_b, str(wav_train))
    wav_valid = wav_dir.join("valid.wav")
    make_wav(data_c, str(wav_valid))

    label_test = label_dir.join("test.phonemes").write("a")
    label_train = label_dir.join("train.phonemes").write("b")
    label_valid = label_dir.join("valid.phonemes").write("c")

    test_prefixes = tmpdir.join("test_prefixes.txt").write("a")
    train_prefixes = tmpdir.join("train_prefixes.txt").write("b")
    valid_prefixes = tmpdir.join("vaild_prefixes.txt").write("c")

    c = Corpus(
        feat_type='fbank',
        label_type='phonemes',
        tgt_dir=Path(str(tmpdir)),
        labels={"a","b","c"}
    )
    assert c


def test_create_corpus_label_mismatch(tmpdir):
    """Test that creation of a Corpus raises an error when the supplied label set
    does not exactly match those found in the provided data"""
    from persephone.corpus import Corpus
    from persephone.exceptions import LabelMismatchException
    from pathlib import Path

    wav_dir = tmpdir.mkdir("wav")
    label_dir = tmpdir.mkdir("label")

    wav_test = wav_dir.join("test.wav").write("")
    wav_train = wav_dir.join("train.wav").write("")
    wav_valid = wav_dir.join("valid.wav").write("")

    label_test = label_dir.join("test.phonemes").write("a")
    label_train = label_dir.join("train.phonemes").write("b")
    label_valid = label_dir.join("valid.phonemes").write("c")

    # TODO: write prefix files

    with pytest.raises(LabelMismatchException):
        c = Corpus(
            feat_type='fbank',
            label_type='phonemes',
            tgt_dir=Path(str(tmpdir)),
            labels=["1", "2", "3"]
        )

def test_determine_labels_throws():
    """Test that a non existant directory will throw"""
    import pathlib
    from persephone.corpus import determine_labels
    non_existent_path = pathlib.Path("thispathdoesntexist")
    with pytest.raises(FileNotFoundError):
        determine_labels(non_existent_path, "phonemes")


def test_determine_labels(tmpdir): #fs is the fake filesystem fixture
    """test the function that determines what labels exist in a directory"""
    from pathlib import Path

    base_dir = tmpdir
    label_dir = base_dir.mkdir("label")

    test_1_phonemes = 'ɖ ɯ ɕ i k v̩'
    test_1_phonemes_and_tones = 'ɖ ɯ ˧ ɕ i ˧ k v̩ ˧˥'
    test_2_phonemes = 'g v̩ tsʰ i g v̩ k v̩'
    test_2_phonemes_and_tones = 'g v̩ ˧ tsʰ i ˩ g v̩ ˩ k v̩ ˩'

    label_dir.join("test1.phonemes").write(test_1_phonemes)
    label_dir.join("test1.phonemes_and_tones").write(test_1_phonemes_and_tones)
    label_dir.join("test2.phonemes").write(test_2_phonemes)
    label_dir.join("test2.phonemes_and_tones").write(test_2_phonemes_and_tones)

    all_phonemes = set(test_1_phonemes.split(' ')) | set(test_2_phonemes.split(' '))

    from persephone.corpus import determine_labels
    phoneme_labels = determine_labels(Path(str(base_dir)), "phonemes")
    assert phoneme_labels
    assert phoneme_labels == all_phonemes

    all_phonemes_and_tones = set(test_1_phonemes_and_tones.split(' ')) | set(test_2_phonemes_and_tones.split(' '))

    phoneme_and_tones_labels = determine_labels(Path(str(base_dir)), "phonemes_and_tones")
    assert phoneme_and_tones_labels
    assert phoneme_and_tones_labels == all_phonemes_and_tones


def test_data_overlap():
    """Tests that the overlap detection function works as advertised"""
    train = set(["a", "b", "c"])
    valid = set(["d", "e", "f"])
    test = set(["g", "h", "i"])
    from persephone.corpus import ensure_no_set_overlap
    from persephone.exceptions import PersephoneException
    ensure_no_set_overlap(train, valid, test)

    overlap_with_train = set(["a"])
    with pytest.raises(PersephoneException):
        ensure_no_set_overlap(train, valid|overlap_with_train, test)
    overlap_with_valid = set(["d"])
    with pytest.raises(PersephoneException):
        ensure_no_set_overlap(train, valid, test|overlap_with_valid)
    overlap_with_test = set(["g"])
    with pytest.raises(PersephoneException):
        ensure_no_set_overlap(train|overlap_with_test, valid, test)

def test_untranscribed_wavs(tmpdir):
    """test that untranscribed wav files are found"""
    from pathlib import Path
    from persephone.corpus import find_untranscribed_wavs

    wav_dir = tmpdir.mkdir("wav")
    label_dir = tmpdir.mkdir("label")

    wav_untranscribed = wav_dir.join("untranscribed1.wav").write("")

    wav_1 = wav_dir.join("1.wav").write("")
    transcription_1 = label_dir.join("1.phonemes").write("")

    untranscribed_prefixes = find_untranscribed_wavs(Path(str(wav_dir)), Path(str(label_dir)), "phonemes")
    assert untranscribed_prefixes
    assert len(untranscribed_prefixes) == 1
    assert "untranscribed1" in untranscribed_prefixes

    untranscribed_prefixes_phonemes_and_tones = find_untranscribed_wavs(Path(str(wav_dir)), Path(str(label_dir)), "phonemes_and_tones")
    assert untranscribed_prefixes_phonemes_and_tones
    assert len(untranscribed_prefixes_phonemes_and_tones) == 2
    assert "untranscribed1" in untranscribed_prefixes_phonemes_and_tones
    assert "1" in untranscribed_prefixes_phonemes_and_tones

def test_untranscribed_prefixes_from_file(tmpdir):
    """Test that extracting prefixes from an "untranscribed_prefixes.txt" file
    will behave as advertised"""
    from pathlib import Path
    from persephone.corpus import get_untranscribed_prefixes_from_file
    untranscribed_prefix_content = """foo
bar
baz"""
    untranscribed_prefix_file = tmpdir.join("untranscribed_prefixes.txt").write(untranscribed_prefix_content)

    untranscribed_prefixes = get_untranscribed_prefixes_from_file(Path(str(tmpdir)))
    assert untranscribed_prefixes
    assert len(untranscribed_prefixes) == 3
    assert "foo" in untranscribed_prefixes
    assert "bar" in untranscribed_prefixes
    assert "baz" in untranscribed_prefixes
