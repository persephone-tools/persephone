API
===

Fundamental classes
-------------------

.. autoclass:: persephone.utterance.Utterance

Preprocessing
-------------

.. autofunction:: persephone.preprocess.elan.utterances_from_dir
.. autofunction:: persephone.preprocess.wav.extract_wavs

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
