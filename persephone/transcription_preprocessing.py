"""
Offers functions for tokenizing utterances into phonemes, characters or
other symbols.
"""

from typing import Iterable

def segment_into_chars(utterance: str) -> str:
    """ Segments an utterance into space delimited characters. """

    if type(utterance) != str:
        raise TypeError("Input type must be a string.")

    utterance.strip()
    utterance = utterance.replace(" ", "")
    return " ".join(utterance)

def segment_into_labels(utterance: str, token_inventory: Iterable[str]):
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

    def pop_label(utterance):
        for i in range(maxlen, 0, -1):
            if line[:i] in token_inventory:
                return line[:i], line[i:]

        return

    token_inventory = set(token_inventory)
    # Get the size of the longest token in the inventory
    max_len = len(sorted(list(token_inventory), key=lambda x: len(x))[-1])
