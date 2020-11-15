Installation
============

As much as possible the Persephone library strives to be a Python-only library to make installation as easy as possible.
Due to the nature of processing sound files we have to interact with various utilities that are non-Python.

The Persephone library requires you have Python 3.5 or Python 3.6 installed.
Currently Python 3.7 is not supported because we depend on Tensorflow which currently does not support Python 3.7
(see the `relevant issue thread <https://github.com/tensorflow/tensorflow/issues/17022>`_)

Installation from PyPi
----------------------

The Persephone library is available on PyPi: https://pypi.org/project/persephone/ 

The easiest way to install is via the `pip package manager <https://pip.pypa.io/en/stable/>`_

.. code:: sh

    pip install persephone

External binaries
-----------------

The library depends on a few binaries being installed:

* FFMPEG
* SOX
* Kaldi (Optional, required for pitch features support)

There are some bootstrap scripts that are used to provision a development environment which will install the required system packages from apt.

See the `Configuration`_ section for how to configure Persephone to use these binaries.

Configuration
-------------

The library requires various binaries to be available and directories to be present in order to work. Various defaults are defined as per values in `persephone/config.py <https://github.com/persephone-tools/persephone/blob/master/persephone/config.py>`_ to override any of these you can create a file called ``settings.ini`` at the same base path that you are invoking Persephone from.

Binaries
~~~~~~~~~~~~

Once you have the binaries in the `External binaries`_ section installed you may need to configure the paths to them.
Here is an example of how to specify the path to required binaries in the ``settings.ini`` file:

.. code::

    [PATHS]
    SOX_PATH = "sox"
    FFMPEG_PATH = "ffmpeg"
    KALDI_ROOT = "/home/oadams/tools/kaldi"


Here "sox" and "ffmpeg" must be available on the path and ``KALDI_ROOT`` specifies an absolute path. Note that these paths can also be specified as absolute paths if you wish.

``KALDI_ROOT_PATH`` can also be used to specify the path to your Kaldi installation but this is deprecated, please use ``KALDI_ROOT`` in your settings file.

Paths
~~~~~

There's a variety of filesystem paths that will be used for storage of data. Here is an example of how to specify paths in the ``settings.ini`` file:

.. code::

    [PATHS]
    CORPORA_BASE_PATH = "./ourdata/original/"
    TGT_DIR = "./preprocessed_data"
    EXP_DIR = "./experiments"

``CORPORA_BASE_PATH`` will specify the base for paths that contain the original un-preprocessed source corpora. The default for this is ``./data/org/``.

``TGT_DIR`` will specify the target directory to store preprocessed data in. The default for this is ``./data``.

``EXP_DIR`` will specify the directory where experiment results are saved in. The default for this is ``./exp``.