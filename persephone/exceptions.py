"""Exceptions that can be raised by the Persephone library are defined here"""

class PersephoneException(Exception):
    """Base class for all exceptions raised by the Persephone library"""

class NoPrefixFileException(PersephoneException):
    """
    Thrown if files like train_prefixes.txt, test_prefixes.txt can't be
    found.
    """

class DirtyRepoException(PersephoneException):
    """
    An exception that is raised if the current working directory
    is in a dirty state according to Git.
    """

class EmptyReferenceException(PersephoneException):
    """
    When calculating word error rates, the reference string must be of 
    length >= 1. Otherwise, this exception will be thrown.
    """
