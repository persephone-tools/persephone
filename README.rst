Persephone v0.3.1 (beta version)
================================

Persephone (/pərˈsɛfəni/) is an automatic phoneme transcription tool.
Traditional speech recognition tools require a large pronunciation
lexicon (describing how words are pronounced) and much training data so
that the system can learn to output orthographic transcriptions. In
contrast, Persephone is designed for situations where training data is
limited, perhaps as little as an hour of transcribed speech. Such
limitations on data are common in the documentation of low-resource
languages. It is possible to use such small amounts of data to train a
transcription model that can help aid transcription, yet such technology
has not been widely adopted.

    The speech recognition tool presented here is named after the
    goddess who was abducted by Hades and must spend one half of each
    year in the Underworld. Which of linguistics or computer science is
    Hell, and which the joyful world of spring and light? For each it’s
    the other, of course. --- Alexis Michaud

The goal of Persephone is to make state-of-the-art phonemic
transcription accessible to people involved in language documentation.
Creating an easy-to-use user interface is central to this. The user
interface and APIs are a work in progress and currently Persephone must
be run via a command line.

The tool is implemented in Python/Tensorflow with extensibility in mind.
Currently just one model is implemented, which uses bidirectional long
short-term memory (LSTMs) and the connectionist temporal classification
(CTC) loss function.

We are happy to offer direct help to anyone who wants to use it. If
you're having trouble, contact Oliver Adams at oliver.adams@gmail.com.
We are also very welcome to thoughts, constructive criticism, help with
design, development and documentation, along with any bug reports or
pull requests you may have.

Documentation
=============

Documentation can be found `here <http://persephone.readthedocs.io/en/latest/>`_.

Contributors
============

Persephone has been built based on the code contributions of:

* Oliver Adams
* `Janis Lesinskis <https://www.customprogrammingsolutions.com/>`_
* Ben Foley
* Nay San

Citation
========

If you use this code in a publication, please cite `Evaluating Phonemic
Transcription of Low-Resource Tonal Languages for Language
Documentation <https://halshs.archives-ouvertes.fr/halshs-01709648/document>`_:

::

    @inproceedings{adams18evaluating,
    title = {Evaluating phonemic transcription of low-resource tonal languages for language documentation},
    author = {Adams, Oliver and Cohn, Trevor and Neubig, Graham and Cruz, Hilaria and Bird, Steven and Michaud, Alexis},
    booktitle = {Proceedings of LREC 2018},
    year = {2018}
    }
