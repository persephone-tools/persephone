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
def create_test_corpus(tmpdir, create_note_sequence, make_wav):
    """Creates a minimal test corpus that doesn't require any files as dependencies"""
    from pathlib import Path
    def _create_corpus():
        from persephone.corpus import Corpus

        wav_dir = tmpdir.mkdir("wav")
        label_dir = tmpdir.mkdir("label")

        #create sine wave data
        data_a = create_note_sequence(notes=["A"])
        data_b = create_note_sequence(notes=["B"])
        data_c = create_note_sequence(notes=["C"])
        data_a_b = create_note_sequence(notes=["A","B"])
        data_b_c = create_note_sequence(notes=["B","C"])
        data_a_b_c = create_note_sequence(notes=["A","B","C"])
        data_c_b_a = create_note_sequence(notes=["C","B","A"])

        #testing
        wav_test1 = wav_dir.join("test1.wav")
        make_wav(data_a_b, str(wav_test1))
        label_test1 = label_dir.join("test1.phonemes").write("A B")

        wav_test2 = wav_dir.join("test2.wav")
        make_wav(data_c, str(wav_test2))
        label_test2 = label_dir.join("test2.phonemes").write("C")

        #training
        wav_train1 = wav_dir.join("train1.wav")
        make_wav(data_b_c, str(wav_train1))
        label_train1 = label_dir.join("train1.phonemes").write("B C")

        wav_train2 = wav_dir.join("train2.wav")
        make_wav(data_a_b_c, str(wav_train2))
        label_train2 = label_dir.join("train2.phonemes").write("A B C")

        wav_train3 = wav_dir.join("train3.wav")
        make_wav(data_a, str(wav_train2))
        label_train3 = label_dir.join("train3.phonemes").write("A")

        wav_train4 = wav_dir.join("train4.wav")
        make_wav(data_b, str(wav_train2))
        label_train4 = label_dir.join("train4.phonemes").write("B")

        wav_train5 = wav_dir.join("train5.wav")
        make_wav(data_c_b_a, str(wav_train5))
        label_train5 = label_dir.join("train5.phonemes").write("C B A")

        #validation
        wav_valid = wav_dir.join("valid.wav")
        make_wav(data_c, str(wav_valid))

        label_valid = label_dir.join("valid.phonemes").write("C")

        # Prefixes handling
        test_prefixes = tmpdir.join("test_prefixes.txt").write("test1\ntest2")
        train_prefixes = tmpdir.join("train_prefixes.txt").write("train1\ntrain2")
        valid_prefixes = tmpdir.join("valid_prefixes.txt").write("valid")

        c = Corpus(
            feat_type='fbank',
            label_type='phonemes',
            tgt_dir=Path(str(tmpdir)),
            labels={"A","B","C"}
        )
        assert c
        assert c.feat_type == 'fbank'
        assert c.label_type == 'phonemes'
        assert set(c.labels) == {"A", "B", "C"}
        assert c.vocab_size == 3
        return c
    return _create_corpus
