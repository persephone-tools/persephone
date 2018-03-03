import random
import string

import pytest
from nltk.metrics import distance

from persephone.distance import min_edit_distance

def rand_str(length: int, alphabet: str = string.digits) -> str:
    return ''.join(random.choice(alphabet) for i in range(length))

@pytest.fixture
def seq_cases():
    hardcoded_seqs = [("", "", 0),
                      ("abde", "abcde", 1),
                      ([1,3,5], [], 3),
                      ([1,3,5], [3], 2),
                     ]

    # Here we assume the nltk.metrics.distance implementation is correct.
    generated_seqs = []
    for length in range(25):
        for _ in range(10):
            s1 = rand_str(length)
            s2 = rand_str(length)
            dist = distance.edit_distance(s1, s2)
            generated_seqs.append((s1, s2, dist))

    return hardcoded_seqs + generated_seqs

def test_med1(seq_cases):

    for s1, s2, dist in seq_cases:
        assert min_edit_distance(s1, s2) == dist
