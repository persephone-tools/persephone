API
===

Fundamental classes
-------------------

.. autoclass:: persephone.utterance.Utterance

.. autoclass:: persephone.corpus.Corpus
   :members:

.. autoclass:: persephone.corpus.ReadyCorpus
   :members:

.. autoclass:: persephone.corpus_reader.CorpusReader
   :members:

.. autoclass:: persephone.model.Model
   :members:

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
