"""Test fixtures setup for pytest

https://docs.pytest.org/en/latest/fixture.html
"""
import pytest
from typing import List

@pytest.fixture
def make_wav():
    """Fixture for convenience in creating real WAV files for tests that require them.
    This is necessary for functionality that requires WAV files such as FFMPEG
    calls or other signal processing tasks where mocking would be far more of a
    pain than just making a file."""
    import wave
    import struct
    import io
    def _make_audio(audio_data, filename: str, framerate: float=44100.00, duration: float=1) -> None:
        """Create a file with appropriate WAV magic bytes and encoding

        :audio_data: raw frame data to be placed into the wav file
        :filename: the filename that will be uploaded
        :framerate: hertz
        :duration: seconds this file will go for
        """
        amp = 8000.0 # amplitude
        # wav params
        nchannels = 1
        sampwidth = 2
        framerate = int(framerate)
        nframes = int(framerate*duration)
        comptype = "NONE"
        compname = "not compressed"

        with wave.open(filename, "wb") as wav_file:
            wav_file.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
            # write the contents
            for s in audio_data:
                wav_file.writeframes(struct.pack('h', int(s*amp/2)))

    return _make_audio


@pytest.fixture
def create_sine():
    def _create_sine(note: str="A", seconds: float=1, framerate: float=44100.00) -> list:
        """Create a sine wave representing the frequency of a piano note
        in octave 4 using the A440 tuning"""
        import math

        note_to_freq = {
            "C": 261.63,
            "C♯": 277.18, "D♭": 277.18,
            "D": 293.66,
            "E♭": 311.13, "D♯": 311.13,
            "E": 329.63,
            "F": 349.23,
            "F♯": 369.99, "G♭": 369.99,
            "G": 392.00,
            "A♭": 415.30, "G♯": 415.30,
            "A": 440.00,
            "B♭": 466.16, "A♯": 466.16,
            "B": 493.88,
        }
        datasize = int(seconds * framerate)
        freq = note_to_freq[note]
        sine_list=[]
        for x in range(datasize):
            sine_list.append(math.sin(2*math.pi * freq * ( x/framerate)))
        return sine_list
    return _create_sine

@pytest.fixture
def create_note_sequence(create_sine):
    """Create the raw data for creating a sequence of notes that can be stored in a WAV file"""
    def _create_note_sequence(notes: List[str], seconds: float=1, framerate: float=44100.0):
        """Create the data for a sequence of pure sine waves that correspond to notes."""
        note_duration = seconds / len(notes)
        data = []
        for note in notes:
            data.extend(
                create_sine(note=note, seconds=note_duration, framerate=framerate))
        return data
    return _create_note_sequence

@pytest.fixture
def create_test_corpus(tmpdir, create_sine, make_wav):
    """Creates a minimal test corpus that doesn't require any files as dependencies"""
    from pathlib import Path
    def _create_corpus():
        from persephone.corpus import Corpus

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

        test_prefixes = tmpdir.join("test_prefixes.txt").write("test")
        train_prefixes = tmpdir.join("train_prefixes.txt").write("train")
        valid_prefixes = tmpdir.join("valid_prefixes.txt").write("valid")

        c = Corpus(
            feat_type='fbank',
            label_type='phonemes',
            tgt_dir=Path(str(tmpdir)),
            labels={"a","b","c"}
        )
        assert c
        assert c.feat_type == 'fbank'
        assert c.label_type == 'phonemes'
        assert set(c.labels) == {"a", "b", "c"}
        assert c.vocab_size == 3
        return c
    return _create_corpus