from pathlib import Path
from typing import List, NamedTuple

Utterance = NamedTuple("Utterance", [("media_path", Path),
                                     ("org_transcription_path", Path),
                                     ("prefix", str),
                                     ("start_time", int),
                                     ("end_time", int),
                                     ("text", str),
                                     ("participant", str)])

def write_utters(utterances: List[Utterance],
                 tgt_dir: Path, ext: str) -> None:
    """ Write the Utterance.text to a file in the tgt_dir. """

    tgt_dir.mkdir()
    for utter in utterances:
        out_path = tgt_dir / "{}.{}".format(utter.prefix, ext)
        with out_path.open() as f:
            print(utter.text, file=f)
