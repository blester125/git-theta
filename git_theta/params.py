"""Classes for serializing model updates."""

import json
import sys
from abc import ABCMeta, abstractmethod
from typing import Optional

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

import msgpack
import tensorstore as ts

from git_theta import utils


class TensorSerializer(metaclass=ABCMeta):
    """Serialize/Deserialize tensors."""

    @abstractmethod
    async def serialize(self, tensor):
        """Convert a tensor to bytes."""

    @abstractmethod
    async def deserialize(self, serialized_tensor):
        """Convert bytes to a tensor object."""


class TensorStoreSerializer(TensorSerializer):
    async def serialize(self, tensor):
        store = await ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "memory"},
                "metadata": {"shape": tensor.shape, "dtype": tensor.dtype.str},
                "create": True,
            },
        )
        await store.write(tensor)
        serialized_param = {
            k.decode("utf-8"): store.kvstore[k] for k in await store.kvstore.list()
        }
        return serialized_param

    async def deserialize(self, serialized_tensor):
        ctx = ts.Context()
        kvs = await ts.KvStore.open("memory://", context=ctx)
        for name, contents in serialized_tensor.items():
            kvs[name] = contents

        store = await ts.open({"driver": "zarr", "kvstore": "memory://"}, context=ctx)
        param = await store.read()
        return param


class JsonSerializer(TensorSerializer):
    async def serialize(self, tensor):
        return json.dumps(tensor)

    async def deserialize(self, serialized_tensor):
        return json.loads(serialized_tensor)


class FileCombiner(metaclass=ABCMeta):
    """Combine and Split serialized tensors, enables single blob processing for multiple tensors."""

    @abstractmethod
    def combine(self, files):
        """Combine multiple byte steams into one."""

    @abstractmethod
    def split(self, file):
        """Split a combined byte stream into original bytes."""


class MsgPackCombiner(FileCombiner):
    def combine(self, files):
        return msgpack.packb(files, use_bin_type=True)

    def split(self, file):
        return msgpack.unpackb(file, raw=False)


class Serializer(metaclass=ABCMeta):
    """Serialize/Deserialize parameters, even when represented with multiple tensors."""

    @abstractmethod
    async def serialize(self, params):
        """Serialize parameter."""

    @abstractmethod
    async def deserialize(self, serialized):
        """Deserialize parameter."""


class UpdateSerializer(Serializer):
    def __init__(self, tensor_serializer, file_combiner):
        self.serializer = tensor_serializer
        self.combiner = file_combiner

    async def serialize(self, params):
        serialized_params = {
            name: await self.serializer.serialize(param)
            for name, param in params.items()
        }
        return self.combiner.combine(serialized_params)

    async def deserialize(self, serialized):
        serialized_params = self.combiner.split(serialized)
        update_params = {
            name: await self.serializer.deserialize(serialized_param)
            for name, serialized_param in serialized_params.items()
        }
        return update_params


def get_update_serializer(serializer_type: Optional[str] = None) -> UpdateSerializer:
    serializer_type = serializer_type or utils.EnvVarConstants.SERALIZER_TYPE
    discovered_plugins = entry_points(group="git_theta.plugins.serializers")
    seralizer = discovered_plugins[serializer_type].load()
    return UpdateSerializer(seralizer(), MsgPackCombiner())
