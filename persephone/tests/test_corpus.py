"""Tests for corpus related items"""
import pytest

from pyfakefs.pytest_plugin import fs

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
