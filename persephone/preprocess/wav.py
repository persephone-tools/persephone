""" Provide functions for preprocessing the WAV files. """

from pathlib import Path
import subprocess
from typing import List

from pydub import AudioSegment # type: ignore

from .. import config
from ..utterance import Utterance

def millisecs_to_secs(millisecs: int) -> int:
    return millisecs / 1000

def trim_wav_ms(in_path: Path, out_path: Path,
                start_time: int, end_time: int) -> None:
    """ Tries to trim a wav with sox, then backs off to pydub/ffmpeg. """

    try:
        trim_wav_sox(in_path, out_path, start_time, end_time)
    except FileNotFoundError:
        trim_wav_pydub(in_path, out_path, start_time, end_time)

def trim_wav_pydub(in_path: Path, out_path: Path,
                start_time: int, end_time: int) -> None:
    """ Crops the wav file. """
    # TODO add logging here
    #print("in_fn: {}".format(in_fn))
    #print("out_fn: {}".format(out_fn))
    in_ext = in_path.suffix[1:]
    out_ext = out_path.suffix[1:]
    audio = AudioSegment.from_file(str(in_path), in_ext)
    trimmed = audio[start_time:end_time]
    # pydub evidently doesn't actually use the parameters when outputting wavs,
    # since it doesn't use FFMPEG to deal with outputtting WAVs. This is a bit
    # of a leaky abstraction. No warning is given, so normalization to 16Khz
    # mono wavs has to happen later. Leaving the parameters here in case it
    # changes
    trimmed.export(str(out_path), format=out_ext,
                   parameters=["-ac", "1", "-ar", "16000"])

def trim_wav_sox(in_path: Path, out_path: Path,
                 start_time: int, end_time: int) -> None:
    """ Crops the wav file at in_fn so that the audio between start_time and
    end_time is output to out_fn. Measured in milliseconds.
    """
    start_time = millisecs_to_secs(start_time)
    end_time = millisecs_to_secs(end_time)
    if not out_path.is_file():
        args = [config.SOX_PATH, str(in_path), str(out_path),
                "trim", str(start_time), "=" + str(end_time)]
        # TODO Use logging here
        print(args[1:])
        subprocess.run(args)

def extract_wavs(utterances: List[Utterance], tgt_dir: Path) -> None:
    """
    Extracts WAVs from the media files associated with a list of utterances
    and puts them in the tgt_dir.
    """
    # TODO Add logging here
    tgt_dir.mkdir(parents=True, exist_ok=True)
    for utter in utterances:
        wav_fn = "{}.{}".format(utter.prefix, "wav")
        out_wav_path = tgt_dir / wav_fn
        if not out_wav_path.is_file():
            trim_wav_ms(utter.media_path, out_wav_path,
                        utter.start_time, utter.end_time)
