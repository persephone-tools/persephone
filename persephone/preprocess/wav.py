""" Provide functions for preprocessing the WAV files. """

import logging
from pathlib import Path
import subprocess
from typing import List

from pydub import AudioSegment # type: ignore

from .. import config
from ..utterance import Utterance

logging.config.fileConfig(config.LOGGING_INI_PATH)

def millisecs_to_secs(millisecs: int) -> float:
    return millisecs / 1000

def trim_wav_ms(in_path: Path, out_path: Path,
                start_time: int, end_time: int) -> None:
    """ Extracts part of a WAV File.

    First attempts to call sox. If sox is unavailable, it backs off to
    pydub+ffmpeg.

    Args:
        in_path: A path to the source file to extract a portion of
        out_path: A path describing the to-be-created WAV file.
        start_time: The point in the source WAV file at which to begin
            extraction.
        end_time: The point in the source WAV file at which to end extraction.

    """

    try:
        trim_wav_sox(in_path, out_path, start_time, end_time)
    except FileNotFoundError:
        # Then sox isn't installed, so use pydub/ffmpeg
        trim_wav_pydub(in_path, out_path, start_time, end_time)
    except subprocess.CalledProcessError:
        # Then there is an issue calling sox. Perhaps the input file is an mp4
        # or some other filetype not supported out-of-the-box by sox. So we try
        # using pydub/ffmpeg.
        trim_wav_pydub(in_path, out_path, start_time, end_time)

def trim_wav_pydub(in_path: Path, out_path: Path,
                start_time: int, end_time: int) -> None:
    """ Crops the wav file. """

    logging.info(
        "Using pydub/ffmpeg to create {} from {}".format(out_path, in_path) +
        " using a start_time of {} and an end_time of {}".format(start_time,
                                                                 end_time))

    if out_path.is_file():
        return

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

    if out_path.is_file():
        return

    start_time_secs = millisecs_to_secs(start_time)
    end_time_secs = millisecs_to_secs(end_time)
    args = [config.SOX_PATH, str(in_path), str(out_path),
            "trim", str(start_time_secs), "=" + str(end_time_secs)]
    # TODO Use logging here
    subprocess.run(args, check=True)

def extract_wavs(utterances: List[Utterance], tgt_dir: Path,
                 lazy: bool) -> None:
    """ Extracts WAVs from the media files associated with a list of Utterance
    objects and stores it in a target directory.

    Args:
        utterances: A list of Utterance objects, which include information
            about the source media file, and the offset of the utterance in the
            media_file.
        tgt_dir: The directory in which to write the output WAVs.
        lazy: If True, then existing WAVs will not be overwritten if they have
            the same name
    """
    tgt_dir.mkdir(parents=True, exist_ok=True)
    for utter in utterances:
        wav_fn = "{}.{}".format(utter.prefix, "wav")
        out_wav_path = tgt_dir / wav_fn
        if lazy and out_wav_path.is_file():
            logging.info("File {} already exists and lazy == {}; not " \
                         "writing.".format(out_wav_path, lazy))
            continue
        logging.info("File {} does not exist and lazy == {}; creating " \
                     "it.".format(out_wav_path, lazy))
        trim_wav_ms(utter.org_media_path, out_wav_path,
                    utter.start_time, utter.end_time)
