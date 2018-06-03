The Persephone API
==================

In this section we discuss the application program interface (API) exposed by
Persephone. We begin with descriptions of the fundamental classes included in
the tool. Model training pipelines are described by instantiating these
classes. Consider the following example for a preliminary look at how this
works::

   # Create a corpus from data that has already been preprocessed.
   # Among other things, this will divide the corpus into training,
   # validation and test sets.
   from persephone.corpus import Corpus
   corpus = Corpus(feat_type="fbank",
                    label_type="phonemes",
                    tgt_dir="/path/to/preprocessed/data")

   # Create an object that reads the corpus data in batches.
   from persephone.corpus_reader import CorpusReader
   corpus_reader = CorpusReader(corpus, batch_size=64)

   # Create a neural network model (LSTM/CTC model) and train
   # it on the corpus.
   from persephone.rnn_ctc import Model
   model = Model("/path/to/experiment/directory",
                 corpus_reader,
                 num_layers=3,
                 num_hidden=250)
   model.train()

This will train and evaluate a model, storing information related to the
specific experiment in `/path/to/experiment/directory`.

In the next section we take a closer look at the classes that comprise this
example, and reveal additional functionality, such as loading the 
speech and transcriptions from `ELAN
<https://tla.mpi.nl/tools/tla-tools/elan/>`_ files and how preprocessing of the
raw transcription text is specified.

On the horizon, but still to be implemented, is description of these pipelines and
interaction between classes in a way that is compatible with the YAML files of
the `eXtensible Neural Machine Translation toolkit (XNMT)
<https://github.com/neulab/xnmt>`_.

Fundamental classes
-------------------

The four key classes are the `Utterance`, `Corpus`, `CorpusReader`, and `Model`
classes. `Utterance` instances comprise `Corpus` instances, which are loaded by
`CorpusReader` instances and fed into `Model` instances.

.. autoclass:: persephone.utterance.Utterance

.. autoclass:: persephone.corpus.Corpus
   :members: __init__, from_elan

.. There is support for creating Corpus objects from ELAN files::

..   # Create a corpus from ELAN input files.
..   from persephone.corpus import Corpus
..   corpus = Corpus.from_elan(org_dir="/path/to/input/data",
..                             tgt_dir="/path/to/preprocessed/data",
..                             utterance_filter=function_to_call,
..                             label_segmenter=something,
..                             tier_prefixes=("xv", "rf"))

.. autoclass:: persephone.corpus_reader.CorpusReader
   :members: __init__, 

.. autoclass:: persephone.model.Model
   :members: __init__, train, transcribe

Preprocessing
-------------

.. autofunction:: persephone.preprocess.elan.utterances_from_dir
.. autoclass:: persephone.preprocess.labels.LabelSegmenter
.. autofunction:: persephone.preprocess.wav.extract_wavs

Models
------

.. autoclass:: persephone.rnn_ctc.Model
   :members:

Distance measurements
---------------------

.. autofunction:: persephone.distance.min_edit_distance
.. autofunction:: persephone.distance.min_edit_distance_align
.. autofunction:: persephone.distance.word_error_rate

Exceptions
----------

.. autoexception:: persephone.exceptions.PersephoneException
.. autoexception:: persephone.exceptions.NoPrefixFileException
.. autoexception:: persephone.exceptions.DirtyRepoException
.. autoexception:: persephone.exceptions.EmptyReferenceException
