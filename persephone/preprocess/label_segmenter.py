from typing import Callable, Set, NamedTuple

from ..utterance import Utterance

LabelSegmenter = NamedTuple("LabelSegmenter",
                            [("segment_labels", Callable[[Utterance], Utterance]),
                            ("labels", Set[str])])
