Persephone v0.2.0 (beta version)
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

Contributors
============

Persephone has been built based on the code contributions of:

* Oliver Adams
* `Janis Lesinskis <https://www.customprogrammingsolutions.com/>`_
* Ben Foley
* Nay San

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

Quickstart
==========

This guide is written to help you get the tool working on your machine.
We will use a example setup that involves training a phoneme
transcription tool for `Yongning Na <http://lacito.vjf.cnrs.fr/pangloss/languages/Na_en.php>`_.
For this we use a small (even by language documentation standards) sub-sampling
of elicited speech of Yongning Na, a language of Southwestern China.

The example that we will run can be run on most personal computers
without a graphics processing unit (GPU), since I've made the settings
less computationally demanding than it would be for optimal
transcription quality. Ideally you'd have access to a server with more
memory and a GPU, but this isn't necessary.

The code has been tested on Mac and Linux systems. It can be run on
Windows using the Docker container described below.

For now you must open up a terminal to enter commands at the command
line. (The commands below are prefixed with a
``":math:`"``. Don't enter the ``"`"``, just whatever comes afterwards).

1. Installation
---------------

Installation option 1: Using the Docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To simplify setup and system dependencies, a Docker container has been
created. This just requires `Docker to be
installed <https://docs.docker.com/install/>`_. Once you have installed
docker you can fetch our container with:

::

    $ docker pull oadams/persephone

Then run it in interactive mode:

::

    $ docker run -it oadams/persephone

This will place you in an environment where Persephone and its
dependencies have been installed, along with the example Na data.

Installation option 2: A "native" install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure Python 3 is installed.

You will also need to install some system dependencies. For your
convienence we have an install script for dependencies for Ubuntu. To
install the Ubuntu binaries, run ``./ubuntu_bootstrap.sh`` to install
ffmpeg packages. On MacOS we suggest installing via Homebrew with
``brew install ffmpeg``.

We now need to set up a virtual environment and install the library.

::

    $ python3 -m virtualenv -p python3 persephone-venv
    $ source persephone-venv/bin/activate
    $ pip install -U pip
    $ pip install persephone

(This library can be installed system-wide but it is recommended to
install in a virtualenv.)

I've uploaded an example dataset that includes some Yongning Na data
that has already been preprocessed. We'll use this example dataset in
this tutorial. Once we confirm that the software itself is working on
your computer, we can discuss preprocessing of your own data.

Create a working directory for storage of the data and running
experiments:

::

    mkdir persephone-tutorial/
    cd persephone-tutorial/
    mkdir data

Get the data
`here <https://cloudstor.aarnet.edu.au/plus/s/YJXTLHkYvpG85kX/download>`_

Unzip ``na_example_small.zip``. There should now be a directory
``na_example/``, with subdirfectories ``wav/`` and ``label/``. You can
put ``na_example`` anywhere, but for the rest of this tutorial I assume
it is in the working directory:
``persephone-tutorial/data/na_example/``.

2. Training a toy Na model
--------------------------

One way to conduct experiments is to run the code from the iPython
interpreter. Back to the terminal:

::

    $ ipython
    > from persephone import corpus
    > corp = corpus.ReadyCorpus("data/na_example")
    > from persephone import run
    > run.train_ready(corp)

You'll should now see something like:

::

    Number of training utterances: 1024
    Batch size: 16
    Batches per epoch: 64
    2018-01-18 10:30:22.290964: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
    exp_dir ./exp/0, epoch 0
        Batch...0...1...2...3...

The message may vary a bit depending on your CPU but if it says
something like this then training is very likely working. Contact me if
you have any trouble getting to this point, or if you had to deviate
from the above instructions to get to this point.

On the current settings it will train through at least 10 "epochs", very
likely more. If you don't have a GPU then this will take quite a while,
though you should notice it converging in performance within a couple
hours on most personal computers.

After a few epochs you can see how its going by going to opening up
``exp/<experiment_number>/train_log.txt``. This will show you the error
rates on the training set and the held-out validation set. In the
``exp/<experiment_number>/decoded`` subdirectory, you'll see the
validation set reference in ``refs`` and the model hypotheses for each
epoch in ``epoch<epoch_num>_hyps``.

Currently the tool assumes each utterance is in its own audio file, and
that for each utterance in the training set there is a corresponding
transcription file with phonemes (or perhaps characters) delimited by
spaces.

3. Using your own data
----------------------

If you have gotten this far, congratulations! You're now ready to start
using your own data. The example setup we created with the Na data
illustrates a couple key points, including how your data should be
formatted, and how you make the system read that data. In fact, if you
format your data in the same way, you can create your own Persephone
``Corpus`` object with:

.. code:: python

    corp = corpus.ReadyCorpus("<your-corpus-directory>", label_type="extension")

where extension is "txt", "phonemes", "tones", or whatever your file has
after the dot.

If you are using the Docker container then to get data in and out of the
container you need to create a "volume" that shares data between your
computer (the host) and the container. If your data is stored in
``/home/username/mydata`` on your machine and in the container you want
to store it in ``/persephone/mydata`` then run:

::

    docker run -it -v /home/username/mydata:/persephone/mydata oadams/persephone

This is simply an extension of the earlier command to run docker, which
additionally specifies the portal with which data is transferred to and
from the container. If Persephone—abducted by Hades—is the queen of the
underworld, then you might consider this volume to be the gates of hell.

Formatting your data
^^^^^^^^^^^^^^^^^^^^

Interfacing with data is a key bottleneck in useability for speech
recognition systems. Providing a simple and flexible interface to your
data is currently the most important priority for Persephone at the
moment. This is a work in progress.

Current data formatting requirements:

* Audio files are stored in ``<your-corpus>/wav/``. The WAV format is supported. Persephone will automatically convert wavs to be 16bit mono 16000Hz.

* Transcriptions are stored in text files in ``<your-corpus>/label/``

* Each audio file is short (ideally no longer than 10 seconds). There is a script added by Ben Foley, ``persephone/scripts/split_eafs.py``, to split audio files into utterance-length units based on ELAN input files. 

* Each audio file in ``wav/`` has a corresponding transcription file in ``label/`` with the same *prefix* (the bit of the filename before the extension). For example, if there is ``wav/utterance_one.wav`` then there should be ``label/utterance_one.<extension>``. ``<extension>`` can be whatever you want, but it should describe how the labelling is done. For example, if it is phonemic then ``wav/utterance_one.phonemes`` is a meaningful filename.

* Each transcript file includes a space-delimited list of *labels* to the model should learn to transcribe. For example:

  - ``data/na_example/label/crdo-NRU_F4_ACCOMP_PFV.0.phonemes`` contains ``l e dz ɯ z e l e dz ɯ z e``
  - ``data/na_example/label/crdo-NRU_F4_ACCOMP_PFV.0.phonemes_and_tones`` might contain: ``l e ˧ dz ɯ ˥ z e ˩ | l e ˧ dz ɯ ˥ z e ˩``

* Persephone is agnostic to what your chosen labels are. It simply tries to figure out how to map speech to that labelling. These labels can be multiple characters long: the spaces demarcate labels. Labels can be any unicode character(s).

* Spaces are used to delimit the units that the tool predicts. Typically these units are phonemes or tones, however they could also just be orthographic characters (though performance is likely to be a bit lower: consider trying to transcribe "$100"). The model can't tell the difference between digraphs and unigraphs as long as they're tokenized in this format, demarcated with spaces.

If your data observes this format then you can load it via the
``ReadyCorpus`` class. If your data does not observe this format, you
have two options:

1. Do your own separate preprocessing to get the data in this format. If
   you're not a programmer this is probably the best option for you. If
   you have ELAN files, this probably means using
   ``persephone/scripts/split_eaf.py``.
2. Create a Python class that inherits from ``persephone.corpus.Corpus``
   (as does ``ReadyCorpus``) and does all your preprocessing. The API
   (and thus documentation) for this is work in progress, but the key
   point is that ``<corpusobject>.train_prefixes``,
   ``<corpusobject>.valid_prefixes``, and
   ``<corpusobject>.test_prefixes`` are lists of prefixes for the
   relevant subset of the data. For now, look at ``ReadyCorpus`` in
   ``persephone/corpus.py`` for an example. For an example on a full
   dataset, see at ``persephone/datasets/na.py`` (beware: here be
   dragons).

Creating validation and test sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently ``ReadyCorpus`` splits the supplied data into three sets
(training, validation and test) in a 95:5:5 ratio. The training set is
what your model is exposed to during training. Validation is a held-out
set that is used to gauge during training how well the model is
performing. Testing is what is used to quantitatively assess model
performance after training is complete.

When you first load your corpus, ``ReadyCorpus`` randomly allocates
files to each of these subsets. If you'd like to do change the prefixes
of which utterances are in in each set, modify
``<your-corpus>/valid_prefixes.txt`` and
``<your-corpus>/test_prefixes.txt``. The training set consists of all
the available utterances in neither of these text files.

4. Miscellaneous Considerations
-------------------------------

On choosing an appropriate label granularity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Question:

    Suprasegmentals like tone, glottalization, nasalization, and
    length are all phonemic in the language I am using. Do they belong in
    one grouping or separately?

Answer:

I'm wary of making sweeping claims about the best approach to
handle all these sorts of phenomena that will realise themselves
differently between languages, since I'm neither a linguist nor do I
have strong understanding for what features the model will learn each
situation. (Regarding tones, the literature on this is also inconclusive
in general). The best thing is to empirically test both approaches:

1. Having features as part of the phoneme token. For example, a
   nasalized /o/ becomes /õ/.
2. Having a separate token that follows the phoneme. For example, a high
   tone /o˥/ becomes two tokens: /o ˥/.

Since there are many ways you can mix and match these, one consideration
to keep in mind is how much larger the label vocabulary becomes by
merging two tokens into one. You don't want this vocabulary to become
too big because then its harder to learn features common to different
tokens, and the model is less likely to pick the right one even if it's
on the right track. In the case of vowel nasalization, maybe you only
double the number of vowels, so it might be worth having merged tokens
for that. If there are 5 different tones though, you might make that
vowel vocabulary about 5 times bigger by combining them into one token,
so its less likely to be good (though who knows, it might still yield
performance improvements).

5. Saving and loading models; transcribing untranscribed data
-------------------------------------------------------------

So far, the tutorial described how to load a ``Corpus`` object, and
perform training and testing with a single function
``run.train_ready(corpus)``, which hid some details. This section
exposes more of the interface so that you can describe models more
fully, save and load models, and apply it to untranscribed data. I'd
like to hear people's thoughts on this interface.

CorpusReaders and Models
^^^^^^^^^^^^^^^^^^^^^^^^

The ``Corpus`` object (of which ``ReadyCorpus`` is a subclass), is an
object that exposes the files in the corpus (among several other
things). Of relevance here is the ``.get_train_fns()``,
``.get_valid_fns()``, ``.get_test_fns()`` methods, which provide lists
of files in the training, validation and test sets respectively. There
is additionally a ``.get_untranscribed_fns()`` method which returns a
list of files representing speech that has not been transcribed.
``.get_untranscribed_fns()`` fetches prefixes of utterances from
``untranscribed_prefixes.txt``, which you can put in the corpus data
directory (at the same level as the ``feat/`` and ``label/``
subdirectories).

To fetch data from your ``Corpus``, a ``CorpusReader`` is used. The
``CorpusReader`` regulates how much data is to be read from the corpus,
as well as the size of the "batches" that are fed to the model during
training. You create a CorpusReader by feeding it a corpus (here the
example na\_corpus):

.. code:: python

    from persephone import corpus
    na_corpus = corpus.ReadyCorpus("data/na_example/")
    from persephone import corpus_reader
    na_reader = corpus_reader.CorpusReader(na_corpus, num_train=512, batch_size=16)

Here, ``na_reader`` is an interface to the corpus which will read from
the corpus files 512 training utterances, in batches of 16 utterances.
We can now feed data to a ``Model``:

.. code:: python

    from persephone import rnn_ctc
    model = rnn_ctc.Model(exp_dir, na_reader, num_layers=2, hidden_size=250)

where ``exp_dir`` is a directory in which experimental results and
logging will be stored. In creating an ``rnn_ctc.Model`` (recurrent
neural network with a connectionist temporal classification loss
function) we have also specified what corpus to read from, how many
layers there are in the neural network, and the amount of "neurons" in
those layers. We can now train the model with:

.. code:: python

    model.train()

After training, we can transcribe untranscribed data with:

.. code:: python

    model.transcribe()

which depends on ``untranscribed_prefixes.txt`` existing before corpus
creation (though there's no reason why this can't be changed to simply
transcribe the utterances with feature files in ``<data-dir>/feat/``
that don't have corresponding transcriptions in ``<data-dir>/label/``).

During training, the model will store the model that performs best on
the validation set in ``<exp_dir>/model``, across a few different files
prefixed with ``model_best.ckpt``. If you later want to load this model
to transcribe untranscribed data, you create a model with the same
hyperparameters and call ``model.transcribe()`` with the
``restore_model_path`` keyword argument:

.. code:: python

    model = rnn_ctc.Model(<new-exp-dir>, na_reader, num_layers=2, hidden_size=250)
    model.transcribe(restore_model_path="<old-exp-dir>/model/model_best.ckpt")

This will load a previous model and perform transcription with it.
