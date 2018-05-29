""" Miscellaneous functions relating to reporting experimental results."""

from pathlib import Path
from typing import Set, Dict, Tuple, Sequence, List, Union

from collections import Counter
from . import utils

from .distance import min_edit_distance_align

def filter_labels(sent: Sequence[str], labels: Set[str] = None) -> List[str]:
    """ Returns only the tokens present in the sentence that are in labels."""

    if labels:
        return [tok for tok in sent if tok in labels]
    return list(sent)

def filtered_error_rate(hyps_path: Union[str, Path], refs_path: Union[str, Path], labels: Set[str]) -> float:
    """ Returns the error rate of hypotheses in hyps_path against references in refs_path after filtering only for labels in labels.
    """
    if isinstance(hyps_path, Path):
        hyps_path = str(hyps_path)
    if isinstance(refs_path, Path):
        refs_path = str(refs_path)

    with open(hyps_path) as hyps_f:
        lines = hyps_f.readlines()
        hyps = [filter_labels(line.split(), labels) for line in lines]
    with open(refs_path) as refs_f:
        lines = refs_f.readlines()
        refs = [filter_labels(line.split(), labels) for line in lines]

    # For the case where there are no tokens left after filtering.
    only_empty = True
    for entry in hyps:
        if entry is not []:
            only_empty = False
            break # found something so can move on immediately
    if only_empty:
        return -1

    return utils.batch_per(hyps, refs)

def latex_header() -> str:
    """ Produces a LaTeX header for the fmt_latex_*() functions to use. """

    return (r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=0.5in,landscape]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{xcolor}"
\usepackage{polyglossia}
\usepackage{booktabs}
\usepackage{longtable}
\setmainfont[Mapping=tex-text,Ligatures=Common,Scale=MatchLowercase]{Doulos SIL}"
\DeclareRobustCommand{hl}[1]{{\textcolor{red}{#1}}}""")

def fmt_latex_output(hyps: Sequence[Sequence[str]],
                     refs: Sequence[Sequence[str]],
                     prefixes: Sequence[str],
                     out_fn: Path,
                    ) -> None:
    """ Output the hypotheses and references to a LaTeX source file for
    pretty printing.
    """

    alignments_ = [min_edit_distance_align(ref, hyp)
                  for hyp, ref in zip(hyps, refs)]

    with out_fn.open("w") as out_f:
        print(latex_header(), file=out_f)

        print("\\begin{document}\n"
              "\\begin{longtable}{ll}", file=out_f)
        print(r"\toprule", file=out_f)
        for sent in zip(prefixes, alignments_):
            prefix = sent[0]
            alignments = sent[1:]
            print("Utterance ID: &", prefix.strip().replace(r"_", r"\_"), r"\\", file=out_f)
            for i, alignment in enumerate(alignments):
                ref_list = []
                hyp_list = []
                for arrow in alignment:
                    if arrow[0] == arrow[1]:
                        # Then don't highlight it; it's correct.
                        ref_list.append(arrow[0])
                        hyp_list.append(arrow[1])
                    else:
                        # Then highlight the errors.
                        ref_list.append("\\hl{%s}" % arrow[0])
                        hyp_list.append("\\hl{%s}" % arrow[1])
                print("Ref: &", "".join(ref_list), r"\\", file=out_f)
                print("Hyp: &", "".join(hyp_list), r"\\", file=out_f)
            print(r"\midrule", file=out_f)

        print(r"\end{longtable}", file=out_f)
        print(r"\end{document}", file=out_f)

def fmt_error_types(hyps: Sequence[Sequence[str]],
                    refs: Sequence[Sequence[str]]
                   ) -> str:
    """ Format some information about different error types: insertions, deletions and substitutions."""

    alignments = [min_edit_distance_align(ref, hyp)
                  for hyp, ref in zip(hyps, refs)]

    arrow_counter = Counter() # type: Dict[Tuple[str, str], int]
    for alignment in alignments:
        arrow_counter.update(alignment)
    sub_count = sum([count for arrow, count in arrow_counter.items()
                if arrow[0] != arrow[1] and arrow[0] != "" and arrow[1] != ""])
    dels = [(arrow[0], count) for arrow, count in arrow_counter.items()
            if arrow[0] != arrow[1] and arrow[0] != "" and arrow[1] == ""]
    del_count = sum([count for arrow, count in dels])
    ins_count = sum([count for arrow, count in arrow_counter.items()
                if arrow[0] != arrow[1] and arrow[0] == "" and arrow[1] != ""])
    total = sum([count for _, count in arrow_counter.items()])

    fmt_pieces = []
    fmt = "{:15}{:<4}\n"
    fmt_pieces.append(fmt.format("Substitutions", sub_count))
    fmt_pieces.append(fmt.format("Deletions", del_count))
    fmt_pieces.append(fmt.format("Insertions", ins_count))
    fmt_pieces.append("\n")
    fmt_pieces.append("Deletions:\n")
    fmt_pieces.extend(["{:4}{:<4}\n".format(label, count)
                       for label, count in
                       sorted(dels, reverse=True, key=lambda x: x[1])])

    return "".join(fmt_pieces)


def fmt_confusion_matrix(hyps: Sequence[Sequence[str]],
                         refs: Sequence[Sequence[str]],
                         label_set: Set[str] = None,
                         max_width: int = 25) -> str:
    """ Formats a confusion matrix over substitutions, ignoring insertions
    and deletions. """

    if not label_set:
        # Then determine the label set by reading
        raise NotImplementedError()

    alignments = [min_edit_distance_align(ref, hyp)
                  for hyp, ref in zip(hyps, refs)]

    arrow_counter = Counter() # type: Dict[Tuple[str, str], int]
    for alignment in alignments:
        arrow_counter.update(alignment)

    ref_total = Counter() # type: Dict[str, int]
    for alignment in alignments:
        ref_total.update([arrow[0] for arrow in alignment])

    labels = [label for label, count
              in sorted(ref_total.items(), key=lambda x: x[1], reverse=True)
              if label != ""][:max_width]

    format_pieces = []
    fmt = "{:3} "*(len(labels)+1)
    format_pieces.append(fmt.format(" ", *labels))
    fmt = "{:3} " + ("{:<3} " * (len(labels)))
    for ref in labels:
        # TODO
        ref_results = [arrow_counter[(ref, hyp)] for hyp in labels]
        format_pieces.append(fmt.format(ref, *ref_results))

    return "\n".join(format_pieces)

def fmt_latex_untranscribed(hyps: Sequence[Sequence[str]],
                            prefixes: Sequence[str],
                            out_fn: Path) -> None:
    """ Formats automatic hypotheses that have not previously been
    transcribed in LaTeX. """

    hyps_prefixes = list(zip(hyps, prefixes))
    def utter_id_key(hyp_prefix):
        hyp, prefix = hyp_prefix
        prefix_split = prefix.split(".")
        return (prefix_split[0], int(prefix_split[1]))
    hyps_prefixes.sort(key=utter_id_key)

    with out_fn.open("w") as out_f:
        print(latex_header(), file=out_f)
        print("\\begin{document}\n"
              "\\begin{longtable}{ll}", file=out_f)
        print(r"\toprule", file=out_f)
        for hyp, prefix in hyps_prefixes:
            print("Utterance ID: &", prefix.strip().replace(r"_", r"\_"), "\\\\", file=out_f)
            print("Hypothesis: &", hyp, r"\\", file=out_f)
            print("\\midrule", file=out_f)
        print(r"\end{longtable}", file=out_f)
        print(r"\end{document}", file=out_f)
