Installation
============

As much as possible the Persephone library strives to be a Python-only library to make installation as easy as possible.
Due to the nature of processing sound files we have to interact with various utilities that are non-Python.

The Persephone library requires you have Python 3.5 or Python 3.6 installed.
Currently Python 3.7 is not supported because we depend on Tensorflow which currently does not suport Python 3.7
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