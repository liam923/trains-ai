import dataclasses
from typing import (
    NoReturn,
    TYPE_CHECKING,
    TypeVar,
    Type,
)


def assert_never(x: NoReturn) -> NoReturn:
    """
    Use this method to check that matching on a Union is exhaustive
    """
    assert False, "Unhandled type: {}".format(type(x).__name__)


if TYPE_CHECKING:
    # fix for @cache with mypy
    # see https://github.com/python/mypy/issues/5107#issuecomment-529372406
    F = TypeVar("F")

    def cache(f: F) -> F:
        return f


else:
    import functools

    cache = functools.cache


_T = TypeVar("_T")


def add_slots(cls: Type[_T]) -> Type[_T]:
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if "__slots__" in cls.__dict__:
        raise TypeError(f"{cls.__name__} already specifies __slots__")

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    cls_dict["__slots__"] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop("__dict__", None)
    # And finally create the class.
    qualname = getattr(cls, "__qualname__", None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)  # type: ignore
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls
