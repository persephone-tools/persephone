"""Exceptions that can be raised by the Persephone library are defined here"""

class PersephoneException(Exception):
    """Base class for all exceptions raised by the Persephone library"""

class DirtyRepoException(Exception):
    """
    An exception that is raised if the current working directory
    is in a dirty state according to Git.
    """
