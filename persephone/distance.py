from typing import Any, Callable, List, Sequence, TypeVar, Tuple

import numpy as np

from persephone.exceptions import EmptyReferenceException

T = TypeVar("T")

def min_edit_distance(
        source: Sequence[T], target: Sequence[T],
        ins_cost: Callable[..., int] = lambda _x: 1,
        del_cost: Callable[..., int] = lambda _x: 1,
        sub_cost: Callable[..., int] = lambda x, y: 0 if x == y else 1) -> int:
    """Finds the minimum edit distance between two sequences using the
    Levenshtein weighting as a default, but offers keyword arguments to supply
    functions to measure the costs for editing with different elements.

    ins_cost -- A function describing the cost of inserting a given char
    del_cost -- A function describing the cost of deleting a given char
    sub_cost -- A function describing the cost of substituting one char for
    another

    """

    # Initialize an m+1 by n+1 array. Note that the strings start from index 1,
    # with index 0 being used to denote the empty string.
    n = len(target)
    m = len(source)
    distance = np.zeros((m+1, n+1), dtype=np.int16)

    # Initialize the zeroth row and column to be the distance from the empty
    # string.
    for i in range(1, m+1):
        distance[i, 0] = distance[i-1, 0] + ins_cost(source[i-1])
    for j in range(1, n+1):
        distance[0, j] = distance[0, j-1] + ins_cost(target[j-1])

    # Do the dynamic programming to fill in the matrix with the edit distances.
    for j in range(1, n+1):
        for i in range(1, m+1):
            distance[i, j] = min(
                    distance[i-1, j] + ins_cost(source[i-1]),
                    distance[i-1, j-1] + sub_cost(source[i-1],target[j-1]),
                    distance[i, j-1] + del_cost(target[j-1]))

    return int(distance[len(source), len(target)])

# This could work on generic sequences but for now it relies on empty strings.
def min_edit_distance_align(
        source: Sequence[str], target: Sequence[str],
        ins_cost: Callable[..., int] = lambda _x: 1,
        del_cost: Callable[..., int] = lambda _x: 1,
        sub_cost: Callable[..., int] = lambda x, y: 0 if x == y else 1
        ) -> List[Tuple[str,str]]:
    """Finds a minimum cost alignment between two strings using the
    Levenshtein weighting as a default, but offering keyword arguments to
    supply functions to measure the costs for editing with different
    characters.

    ins_cost -- A function describing the cost of inserting a given char
    del_cost -- A function describing the cost of deleting a given char
    sub_cost -- A function describing the cost of substituting one char for
    another

    """

    print("source: ", repr(source))
    print("target: ", repr(target))

    # Initialize an m+1 by n+1 array to hold the distances, and an equal sized
    # array to store the backpointers. Note that the strings start from index
    # 1, with index 0 being used to denote the empty string.
    n = len(target)
    m = len(source)
    dist = [[0]*(n+1) for _ in range(m+1)]
    bptrs = [[[]]*(n+1) for _ in range(m+1)] # type: List[List[List]]

    # Adjust the initialization of the first column and row of the matrices to
    # their appropriate values.
    for i in range(1, m+1):
        dist[i][0] = i
        bptrs[i][0] = (i-1, 0)
    for j in range(1, n+1):
        dist[0][j] = j
        bptrs[0][j] = (0, j-1)

    # Do the dynamic programming to fill in the matrix with the edit distances.
    for j in range(1, n+1):
        for i in range(1, m+1):
            options = [
                    (dist[i-1][j] + ins_cost(target[j-1]),
                        (i-1, j)),
                    (dist[i-1][j-1] + sub_cost(source[i-1],target[j-1]),
                        (i-1, j-1)),
                    (dist[i][j-1] + del_cost(source[i-1]),
                        (i, j-1))]
            (minimum, pointer) = sorted(options)[0]
            dist[i][j] = minimum
            bptrs[i][j] = pointer

    print(bptrs)

    # Put the backtrace in a list, and reverse it to get a forward trace.
    bt = [(m,n)]
    cell = bptrs[m][n]
    print(cell)
    while True:
        if not cell:
            # If it's an empty list or tuple, as will be the case at the start
            # of the trace
            break
        bt.append(cell)
        if bptrs[cell[0]][cell[1]]:
            cell = bptrs[cell[0]][cell[1]]
        else:
            break
    trace = list(reversed(bt))
    print("Trace", trace)

    # Construct an alignment between source and target using the trace.
    alignment = []
    for i in range(1, len(trace)):
        current = trace[i]
#        if not current:
#            # If it's an emtpy list or tuple, as will be the case at the start
#            # of the trace
#            continue
        prev = trace[i-1]

        print("Current: ", current)
        print("Prev: ", prev)

        # If the cell is diagonal from the previous one, it's a substitution or
        # there wasn't a change.
        if current[0] - prev[0] == 1 and current[1] - prev[1] == 1:
            alignment.append((source[current[0]-1], target[current[1]-1]))
        # Otherwise if it moves only on the source side, it's a deletion
        elif current[0] - prev[0] == 1:
            alignment.append((source[current[0]-1], ""))
        # Otherwise if it moves only on the target side, it's an insertion
        elif current[1] - prev[1] == 1:
            alignment.append(("", target[current[1]-1]))

    print("Alignment: ", alignment)
    return alignment

def cluster_alignment_errors(alignment):
    """Takes an alignment created by min_edit_distance_align() and groups
    consecutive errors together. This is useful, because there are often
    many possible alignments, and so often we can't meaningfully distinguish
    between alignment errors at the character level, so it makes many-to-many
    mistakes more readable."""

    newalign = []
    mistakes = ([],[])
    for align_item in alignment:
        if align_item[0] == align_item[1]:
            if mistakes != ([],[]):
                newalign.append((tuple(mistakes[0]), tuple(mistakes[1])))
                mistakes = ([],[])
            newalign.append((tuple([align_item[0]]), tuple([align_item[1]])))
        else:
            if align_item[0] != "":
                mistakes[0].append(align_item[0])
            if align_item[1] != "":
                mistakes[1].append(align_item[1])
    if mistakes != ([],[]):
        newalign.append((tuple(mistakes[0]), tuple(mistakes[1])))
        mistakes = ([],[])

    return newalign

def word_error_rate(ref: Sequence[T], hypo: Sequence[T]) -> float:
    """ Returns the word error rate of the supplied hypothesis with respect to
    the reference string.
    """

    if len(ref) == 0:
        raise EmptyReferenceException(
            "Cannot calculating word error rate against a length 0 "\
            "reference sequence.")

    distance = min_edit_distance(ref, hypo)
    return 100 * float(distance) / len(ref)
