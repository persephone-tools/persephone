"""Tests for corpus related items"""
import pytest

#from pyfakefs.pytest_plugin import fs

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

@pytest.mark.skip("This currently fails because of a bug in pyfakefs,"
                  "see https://github.com/jmcgeheeiv/pyfakefs/issues/409")
def test_determine_labels(fs): #fs is the fake filesystem fixture
    """test the function that determines what labels exist in a directory"""
    from pyfakefs.fake_filesystem_unittest import Patcher

    with Patcher(use_dynamic_patch=True) as patcher:
        import pathlib
    base_dir = pathlib.Path('/tmp/corpus_data')
    label_dir = base_dir / "label"
    fs.create_dir(str(base_dir))
    fs.create_dir(str(label_dir))
    test_1_phonemes = 'ɖ ɯ ɕ i k v̩'
    test_1_phonemes_and_tones = 'ɖ ɯ ˧ ɕ i ˧ k v̩ ˧˥'
    test_2_phonemes = 'g v̩ tsʰ i g v̩ k v̩'
    test_2_phonemes_and_tones = 'g v̩ ˧ tsʰ i ˩ g v̩ ˩ k v̩ ˩'
    fs.create_file(str(label_dir / "test1.phonemes"), contents=test_1_phonemes)
    fs.create_file(str(label_dir / "test1.phonemes_and_tones"), contents=test_1_phonemes_and_tones)
    fs.create_file(str(label_dir / "test2.phonemes"), contents=test_2_phonemes)
    fs.create_file(str(label_dir / "test2.phonemes_and_tones"), contents=test_1_phonemes_and_tones)
    assert base_dir.exists()
    assert label_dir.exists()

    from persephone.corpus import determine_labels
    phoneme_labels = determine_labels(base_dir, "phonemes")
    assert phoneme_labels

    phoneme_and_tones_labels = determine_labels(base_dir, "phonemes_and_tones")
    assert phoneme_and_tones_labels


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