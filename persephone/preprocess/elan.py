""" Provides a Corpus class that can read ELAN .eaf XML files. """

from .. import corpus

class Corpus(corpus.Corpus):
    def __init__(self, tgt_dir, feat_type="fbank", label_type="phonemes",
                 label_segmenter=character_segmenter):
        """ Need to think about this constructor. Ideally it should take a
        function:

            label_preprocess(utter: String) -> String

        which takes a unpreprocessed utterance (perhaps with spaces delimiting
        words in some orthography), and outputs a string where spaces delimit
        things like phonemes and tones.

        corpus.Corpus and corpus.ReadyCorpus could also take such an argument.

        There's probably also going to need to be a way for such a function to
        mark utterances so that they aren't included in the corpus. For
        example, for filtering out code-switched sentences.

        Currently the corpus.Corpus superclass takes a labels argument which is
        just a collection of phonemes etc which the corpus assumes have already
        been segmented correctly, so that it doesn't have to read the whole
        corpus to figure them out. corpus.Corpus should also have the option of
        taking the label_preprocess argument, in which case it would not take
        labels.

        This constructor, elan.Corpus, shouldn't be taking a labels
        argument, since ELAN files are unlikely ever going to
        phoneme-segmented and we don't really want to encourage linguists to do that
        and add tiers or anything. There could be a function that takes labels
        and produces a label_preprocess function that is based on the greedy
        left-to-right phoneme segmentation. So one would do:

            > labels = {"a", "x", "b", ..., etc}
            > greedy_segmenter = create_greedy_segmenter(labels)
            > corp = elan.Corpus(tgt_dir, label_preprocess=greedy_segmenter,
                                 feat_type="fbank", label_type="phonemes")

        I notice now that this means that label_type and label_preprocess needs
        to be manually coordinated by the creater of the elan.Corpus. Ideally 
        the label_preprocess function would dictate the label_type somehow."""
