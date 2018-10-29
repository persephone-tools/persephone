def test_empty_wave_skipped(tmp_path, make_wav):
    """Test that an empty wave file will be skipped instead of crashing."""
    from persephone.preprocess import feat_extract
    wavs_dir = tmp_path / "audio"
    wavs_dir.mkdir()
    no_data = []
    empty_wav_path = wavs_dir / "empty.wav"
    make_wav(no_data, str(empty_wav_path))
    feat_extract.from_dir(wavs_dir, "fbank")