from __future__ import annotations

import inspect
from decimal import Decimal
from enum import Enum
from types import UnionType
from typing import Literal, Union, get_args, get_origin

from pydantic import StrictBool, StrictFloat, StrictInt


def strictify_annotation(annotation: object) -> object:
    if annotation is inspect._empty:
        return str
    if annotation is int:
        return StrictInt
    if annotation is float:
        return StrictFloat
    if annotation is bool:
        return StrictBool
    if annotation is Decimal:
        return Decimal
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is Literal:
        return annotation
    if origin in (UnionType, Union):
        raw_args = get_args(annotation)
        strict_args = tuple(strictify_annotation(arg) for arg in raw_args)
        return Union[strict_args]

    raw_args = get_args(annotation)
    if not raw_args:
        return annotation
    strict_args = tuple(strictify_annotation(arg) for arg in raw_args)
    try:
        return origin[strict_args]
    except TypeError:
        return annotation

