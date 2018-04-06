"""
Offers functions for tokenizing utterances into phonemes, characters or
other symbols.
"""

from typing import Callable, Iterable, NamedTuple, Set, Dict

from ..utterance import Utterance

LabelSegmenter = NamedTuple("LabelSegmenter",
                            [("segment_labels", Callable[[Utterance], Utterance]),
                            ("labels", Set[str])])
LabelSegmenter.__doc__ = (
    """ An immutable object that segments the phonemes of an utterance. This
    could probably actually have a __call__ implementation. That won't work
    because namedtuples can't have special methods. Perhaps it could instead
    just be a function which we give a labels attribute. Perhaps that
    obfuscates things a bit, but it could be okay.

    Attributes:
        segment_labels: A function that takes an Utterance and returns another
            Utterance where the text field has changed to be phonemically
            segmented, using spaces as delimiters. Eg "this is" -> "th i s i s".
        labels: A set of labels (eg. phonemes or tones) relevant for segmenting.
    """)


def segment_into_chars(utterance: str) -> str:
    """ Segments an utterance into space delimited characters. """

    if not isinstance(utterance, str):
        raise TypeError("Input type must be a string. Got {}.".format(type(utterance)))

    utterance.strip()
    utterance = utterance.replace(" ", "")
    return " ".join(utterance)

def segment_into_tokens(utterance: str, token_inventory: Iterable[str]):
    """
    Segments an utterance (a string) into tokens based on an inventory of
    tokens (a list or set of strings).

    The approach: Given the rest of the utterance, find the largest token (in
    character length) that is found in the token_inventory, and treat that as a
    token before segmenting the rest of the string.

    Note: Your orthography may open the door to ambiguities in the
    segmentation. Hopefully not, but another alternative it to simply segment
    on characters with segment_into_chars()
    """

    if not isinstance(utterance, str):
        raise TypeError("Input type must be a string. Got {}.".format(type(utterance)))

    # Token inventory needs to be hashable for speed
    token_inventory = set(token_inventory)
    # Get the size of the longest token in the inventory
    max_len = len(sorted(list(token_inventory), key=lambda x: len(x))[-1])

    def segment_token(utterance):
        if utterance == "":
            return "", ""

        for i in range(max_len, 0, -1):
            if utterance[:i] in token_inventory:
                return utterance[:i], utterance[i:]
        # If the next character is preventing segmentation, move on.
        # TODO This needs to be logged with a warning on the first occurrence.
        return "", utterance[1:]

    tokens = []
    head, tail = segment_token(utterance)
    tokens.append(head)
    while tail != "":
        head, tail = segment_token(tail)
        tokens.append(head)
    tokens = [tok for tok in tokens if tok != ""]

    return " ".join(tokens)

def make_indices_to_labels(labels: Set[str]) -> Dict[int, str]:
    """ Creates a mapping from indices to labels. """

    return {index: label for index, label in
            enumerate(["pad"] + sorted(list(labels)))}
