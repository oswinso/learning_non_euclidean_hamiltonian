import dataclasses
from typing import Any, Dict, Union

import attrs


def is_dc_or_attr(obj: Any) -> bool:
    return is_attr_class(obj) or is_dataclass(obj)


def is_attr_class(obj: Any) -> bool:
    from omegaconf.base import Node

    if attrs is None or isinstance(obj, Node):
        return False
    return attrs.has(obj)


def is_dataclass(obj: Any) -> bool:
    from omegaconf.base import Node

    if dataclasses is None or isinstance(obj, Node):
        return False
    return dataclasses.is_dataclass(obj)


def fields_dict(obj: Any) -> Dict[str, Union[attrs.Attribute, dataclasses.Field]]:
    if is_attr_class(obj):
        return attrs.fields_dict(obj)
    elif is_dataclass(obj):
        return obj.__dataclass_fields__
    else:
        raise ValueError("Trying to get fields dict")


def asdict(obj: Any) -> Dict[str, Any]:
    if is_attr_class(obj):
        return attrs.asdict(obj)
    else:
        return dataclasses.asdict(obj)
