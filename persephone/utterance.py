from typing import NamedTuple

Utterance = NamedTuple("Utterance", [("wav_file", str),
                                     ("transcription_file", str),
                                     ("prefix", str),
                                     ("start_time", int),
                                     ("end_time", int),
                                     ("text", str),
                                     ("participant", str)])

def write_utters(utterances: List[Utterance],
                 tgt_dir: Path, ext: str) -> None:
    """ Write the Utterance.text to a file in the tgt_dir. """

    for utter in utterances:
        out_path = tgt_dir / "{}.{}".format(utter.prefix, ext)
        with out_path.open() as f:
            print(utter.text, file=f)
