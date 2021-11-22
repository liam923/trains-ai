from typing import NoReturn, TYPE_CHECKING, TypeVar


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
    pass
