from collections import defaultdict
from pathlib import Path
from typing import List, NamedTuple, Set, Tuple, DefaultDict

import pint # type: ignore

ureg = pint.UnitRegistry()

Utterance = NamedTuple("Utterance", [("media_path", Path),
                                     ("org_transcription_path", Path),
                                     ("prefix", str),
                                     ("start_time", int),
                                     ("end_time", int),
                                     ("text", str),
                                     ("participant", str)])
Utterance.__doc__= (
    """ Here's a docstring.

    Attributes:
        media_path: blah
        org_transcription_path: blah2

    """)

def write_utters(utterances: List[Utterance],
                 tgt_dir: Path, ext: str) -> None:
    """ Write the Utterance.text to a file in the tgt_dir.

    Args:
        utterances: A list of Utterance objects to be written.
        tgt_dir: The directory in which to write the text of the utterances,
            one file per utterance.
        ext: The file extension for the utterances. Typically something like
            "phonemes", or "phonemes_and_tones".

    """

    tgt_dir.mkdir(parents=True)
    for utter in utterances:
        out_path = tgt_dir / "{}.{}".format(utter.prefix, ext)
        with out_path.open("w") as f:
            print(utter.text, file=f)

def remove_duplicates(utterances: List[Utterance]) -> List[Utterance]:
    """ Removes utterances with the same start_time, end_time and text. Other
    metadata isn't considered.
    """

    filtered_utters = []
    utter_set = set() # type: Set[Tuple[int, int, str]]
    for utter in utterances:
        if (utter.start_time, utter.end_time, utter.text) in utter_set:
            continue
        filtered_utters.append(utter)
        utter_set.add((utter.start_time, utter.end_time, utter.text))

    return filtered_utters

def remove_empty(utterances: List[Utterance]) -> List[Utterance]:
    return [utter for utter in utterances if utter.text.strip() != ""]

def duration(utterances: List[Utterance],
              in_units=ureg.milliseconds, out_units=ureg.minutes) -> int:

    durations = [(utter.end_time - utter.start_time) for utter in utterances]
    total_dur = sum(durations) * in_units
    return total_dur.to(out_units).magnitude

def speaker_durations(utterances: List[Utterance]) -> List[Tuple[str, int]]:
    """ Takes a list of utterances and itemizes them by speaker, returning a
    list of tuples of the form (Speaker Name, duration).
    """

    speaker_utters = defaultdict(list) # type: DefaultDict[str, List[Utterance]]
    for utter in utterances:
        speaker_utters[utter.participant].append(utter)

    speaker_durations = []
    for speaker in speaker_utters:
        speaker_durations.append((speaker, duration(speaker_utters[speaker])))

    return speaker_durations
