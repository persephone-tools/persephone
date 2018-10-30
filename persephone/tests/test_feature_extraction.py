import pytest
def test_empty_wave(tmp_path, create_note_sequence, make_wav):
    """Test that empty wav files are detected"""
    from persephone.preprocess.feat_extract import empty_wav
    wavs_dir = tmp_path / "audio"
    wavs_dir.mkdir()
    no_data = []
    empty_wav_path = wavs_dir / "empty.wav"
    make_wav(no_data, str(empty_wav_path))
    assert empty_wav(empty_wav_path)

    data_a_b = create_note_sequence(notes=["A","B"])
    wav_test1 = wavs_dir / "test1.wav"
    make_wav(data_a_b, str(wav_test1))
    assert not empty_wav(wav_test1)

def test_empty_wave_skipped(tmp_path, make_wav):
    """Test that an empty wave file will be skipped instead of crashing."""
    from persephone.preprocess import feat_extract
    from persephone.exceptions import PersephoneException
    wavs_dir = tmp_path / "audio"
    wavs_dir.mkdir()
    no_data = []
    empty_wav_path = wavs_dir / "empty.wav"
    make_wav(no_data, str(empty_wav_path))
    with pytest.raises(PersephoneException):
        feat_extract.from_dir(wavs_dir, "fbank")