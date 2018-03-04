from collections import defaultdict
import json
from pathlib import Path
from typing import List, NamedTuple, Set, Sequence, Tuple, DefaultDict, Dict

Utterance = NamedTuple("Utterance", [("org_media_path", Path),
                                     ("org_transcription_path", Path),
                                     ("prefix", str),
                                     ("start_time", int),
                                     ("end_time", int),
                                     ("text", str),
                                     ("speaker", str)])
Utterance.__doc__= (
    """ An immutable object that represents a single utterance.

    Attributes:
        org_media_path: A pathlib.Path the original source audio that contains the
            utterance, which may comprise many utterances.
        org_transcription_path: A pathlib.Path to the source of the transcription of
            the utterance, which may comprise many utterances (say, in the
            case of ELAN files).
        prefix: A string identifier for the utterance which is used to prefix the
            target wav and transcription files, which are called <prefix>.wav,
            <prefix>.phonemes, etc.
        start_time: An integer denoting the offset, in milliseconds, of the
            utterance in the original media file.
        end_time: An integer denoting the endpoint, in milliseconds, of the
            utterance in the original media file.
        text: A string representation of the transcription.
        speaker: A string identifier for the speaker of the utterance.

    """)

def write_transcriptions(utterances: List[Utterance],
                         tgt_dir: Path, ext: str, lazy: bool) -> None:
    """ Write the utterance transcriptions to files in the tgt_dir. Is lazy and
    checks if the file already exists.

    Args:
        utterances: A list of Utterance objects to be written.
        tgt_dir: The directory in which to write the text of the utterances,
            one file per utterance.
        ext: The file extension for the utterances. Typically something like
            "phonemes", or "phonemes_and_tones".

    """

    tgt_dir.mkdir(parents=True, exist_ok=True)
    for utter in utterances:
        out_path = tgt_dir / "{}.{}".format(utter.prefix, ext)
        if lazy and out_path.is_file():
            continue
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

def remove_empty_text(utterances: List[Utterance]) -> List[Utterance]:
    return [utter for utter in utterances if utter.text.strip() != ""]

# Doing everything in milliseconds now; other units are only for reporting to
# users
def duration(utter: Utterance) -> int:
    return utter.end_time - utter.start_time

def total_duration(utterances: List[Utterance]) -> int:
    return sum([duration(utter) for utter in utterances])

def make_speaker_utters(utterances: List[Utterance]) -> Dict[str, List[Utterance]]:
    """ Creates a dictionary mapping from speakers to their utterances. """

    speaker_utters = defaultdict(list) # type: DefaultDict[str, List[Utterance]]
    for utter in utterances:
        speaker_utters[utter.speaker].append(utter)

    return speaker_utters

def speaker_durations(utterances: List[Utterance]) -> List[Tuple[str, int]]:
    """ Takes a list of utterances and itemizes them by speaker, returning a
    list of tuples of the form (Speaker Name, duration).
    """

    speaker_utters = make_speaker_utters(utterances)

    speaker_durations = []
    for speaker in speaker_utters:
        speaker_durations.append((speaker, total_duration(speaker_utters[speaker])))

    return speaker_durations

def remove_too_short(utterances: List[Utterance],
                     _winlen=25, winstep=10) -> List[Utterance]:
    """ Removes utterances that will probably have issues with CTC because of
    the number of frames being less than the number of tokens in the
    transcription. Assuming char tokenization to minimize false negatives.
    """
    def is_too_short(utterance: Utterance) -> bool:
        charlen = len(utterance.text)
        if (duration(utterance) / winstep) < charlen:
            return True
        else:
            return False

    return [utter for utter in utterances if not is_too_short(utter)]
