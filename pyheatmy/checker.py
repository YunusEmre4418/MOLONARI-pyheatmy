"""
Module that implements a decorator allowing a method of user-defined class instance to become a "checker".
Also creates a new associated error type.
"""

from functools import wraps


class ComputationOrderException(Exception):
    """Exception raised when a method with a needed tag is computed before the linked checker."""


def checker(checked_meth):
    """
    Transform a method of a user-defined class into a "checker" method: calling it marks the instance, and each
    method decorated with .needed raises a ComputationOrderException on an instance that is not marked.

    The mark is a flag of the instance, so copies and pickles of the instance keep it and their own methods.

    Args:
        checked_meth (function): a method of a user-defined class

    Returns:
            function: checker method
    """
    chkd_name = "__" + checked_meth.__name__

    def needed(meth):
        @wraps(meth)
        def new_meth(self, *args, **kargs):
            if getattr(self, chkd_name, False):
                return meth(self, *args, **kargs)
            raise ComputationOrderException(
                f"{checked_meth.__name__} has to be computed before calling {meth.__name__}."
            )

        return new_meth

    @wraps(checked_meth)
    def wrapper(self, *args, **kwargs):
        setattr(self, chkd_name, True)
        return checked_meth(self, *args, **kwargs)

    wrapper.needed = needed
    return wrapper
