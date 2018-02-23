""" Provide functions for preprocessing the WAV files. """

from pathlib import Path
from typing import List

from pydub import AudioSegment # type: ignore

from ..utterance import Utterance

def trim_wav_ms(in_path: Path, out_path: Path,
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
