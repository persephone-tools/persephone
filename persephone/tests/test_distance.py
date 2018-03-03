import random
import string

import pytest
from nltk.metrics import distance

from persephone.distance import min_edit_distance, min_edit_distance_align
from persephone.distance import word_error_rate
from persephone.exceptions import EmptyReferenceException

def rand_str(length: int, alphabet: str = string.digits) -> str:
    return ''.join(random.choice(alphabet) for i in range(length))

@pytest.fixture
def seq_cases():
    """ Cases are of the form (reference, hypothesis, substitution_cost, dist).
    """
    hardcoded_seqs = [("", "", 1, 0),
                      ("ab", "ad", 1, 1),
                      ("abde", "abcde", 1, 1),
                      ([1,3,5], [], 1, 3),
                      ([1,3,5], [3], 1, 2),
                     ]

    # Here we assume the nltk.metrics.distance implementation is correct.
    generated_seqs = []
    for length in range(25):
        for _ in range(10):
            length2 = random.randint(0, int(length*1.5))
            s1 = rand_str(length)
            s2 = rand_str(length2)
            sub_cost = random.randint(0, 3)
            dist = distance.edit_distance(s1, s2, substitution_cost=sub_cost)
            generated_seqs.append((s1, s2, sub_cost, dist))

    return hardcoded_seqs + generated_seqs

def test_med(seq_cases):

    # Tests edit distance implementation against that of NLTK, with randomized
    # strings and substitution cost functions.
    for s1, s2, sub_cost, dist in seq_cases:
        print(s1, s2, sub_cost, dist)
        sub_cost_f = lambda x, y: 0 if x == y else sub_cost
        assert min_edit_distance(s1, s2, sub_cost=sub_cost_f) == dist
        if len(s1) == 0:
            with pytest.raises(EmptyReferenceException):
                word_error_rate(s1, s2)
        else:
            word_error_rate(s1, s2)

def test_med_align(seq_cases):
    for ref, hyp, sub_cost, dist in seq_cases:
        print("----------------------")
        if not isinstance(ref, str):
            # Ignore generic sequences
            continue
        print(ref, hyp, sub_cost, dist)
        sub_cost_f = lambda x, y: 0 if x == y else sub_cost
        alignment = min_edit_distance_align(ref, hyp, sub_cost=sub_cost_f)
        align_dist = 0
        for arrow in alignment:
            if arrow[0] != arrow[1]:
                if arrow[0] == "": # it's an insertion
                    align_dist += 1
                elif arrow[1] == "": # it's a deletion
                    align_dist += 1
                else: # It's a substitution
                    align_dist += sub_cost
        assert align_dist == dist

# TODO Test the ins_cost and del_cost arguments of min_edit_distance and
# min_edit_distance_align
