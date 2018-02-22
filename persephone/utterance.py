from typing import NamedTuple

Utterance = NamedTuple("Utterance", [("wav_file", str),
                                     ("transcription_file", str),
                                     ("prefix", str),
                                     ("start_time", int),
                                     ("end_time", int),
                                     ("text", str),
                                     ("participant", str)])
