"""Test fixtures setup for pytest

https://docs.pytest.org/en/latest/fixture.html
"""
import pytest

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