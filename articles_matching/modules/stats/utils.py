"""Module with utils for stats calculation"""

from typing import Any, Callable


def ignore_zero_division(func: Callable) -> Callable:
    def inner(*args: Any, **kwargs: Any) -> float:
        res = 0.0

        try:
            res = func(*args, **kwargs)
        except ZeroDivisionError:
            pass

        return res

    return inner
