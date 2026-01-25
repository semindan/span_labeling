from typing import Type


class MethodRegistryMeta(type):
    """Metaclass that auto-registers method classes"""

    _registry: dict[str, Type["SpanLabelerBase"]] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        registry_name = namespace.get("name")
        if registry_name and registry_name != "base":
            mcs._registry[registry_name] = cls
        return cls


class DatasetRegistryMeta(type):
    """Metaclass that auto-registers dataset classes"""

    _registry: dict[str, Type["DatasetBase"]] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        registry_name = namespace.get("key")
        if registry_name:
            mcs._registry[registry_name] = cls
        return cls


class SpanLabelerBase(metaclass=MethodRegistryMeta):
    """Base class for span labeling methods"""

    pass


class DatasetBase(metaclass=DatasetRegistryMeta):
    """Base class for datasets"""

    pass


class MethodRegistry:
    """Access point for registered methods"""

    @classmethod
    def get(cls, name: str) -> Type[SpanLabelerBase]:
        if name not in MethodRegistryMeta._registry:
            raise KeyError(
                f"Method '{name}' not registered. Available: {list(MethodRegistryMeta._registry.keys())}"
            )
        return MethodRegistryMeta._registry[name]

    @classmethod
    def list_all(cls) -> list:
        return list(MethodRegistryMeta._registry.keys())


class DatasetRegistry:
    """Access point for registered datasets"""

    @classmethod
    def get(cls, name: str) -> Type[DatasetBase]:
        if name not in DatasetRegistryMeta._registry:
            raise KeyError(
                f"Dataset '{name}' not registered. Available: {list(DatasetRegistryMeta._registry.keys())}"
            )
        return DatasetRegistryMeta._registry[name]

    @classmethod
    def list_all(cls) -> list:
        return list(DatasetRegistryMeta._registry.keys())
