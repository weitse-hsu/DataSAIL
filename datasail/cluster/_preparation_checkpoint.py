"""Explicit DataSet snapshots at similarity, clustering and pre-solver boundaries.

Uses the Foldseek checkpoint format/lock/publication utilities. This is a narrow
schema for preparation values, never an arbitrary Python object/pickle loader.
"""
import hashlib
import importlib.metadata
import math
import os
import random
import re
import shutil
import struct
from argparse import Namespace
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields
from pathlib import Path

import numpy as np

from datasail.cluster._foldseek_checkpoint import (
    _FORMAT, _VERSION, _fingerprint, _locked, _publish_json, _read_json, _sha, _sync_directory,
)
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER

_ACTIVE = ContextVar("datasail_preparation_checkpoint", default=None)
_STAGES = {"preparation", "cluster-e", "cluster-f", "ligand-e", "ligand-f"}
_SCHEMA = 1


class UnsupportedState(ValueError):
    """Continue ordinary execution without publishing unsupported state."""


def _allowed(name):
    return name in {"owner.json", "owner.json.part", ".lock", "array.part"} or any(
        name == stage + ending for stage in _STAGES for ending in (".json", ".json.part")
    ) or re.fullmatch(r"[0-9a-f]{64}\.npy", name) is not None


def _array_key(array):
    digest = hashlib.sha256(repr((array.dtype.str, array.shape, bool(array.flags.f_contiguous and not array.flags.c_contiguous))).encode())
    raw = memoryview(array.ravel(order="K")).cast("B") if array.nbytes else memoryview(b"")
    for start in range(0, len(raw), 1024 * 1024):
        digest.update(raw[start:start + 1024 * 1024])
    return digest.hexdigest()


def _file_array_key(path):
    # Verify existing blobs in bounded chunks rather than faulting a second
    # whole dense mapping into the worker's RSS. NumPy owns header parsing.
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError("unsupported checkpoint array format")
        shape, fortran, dtype = readers[version](stream)
        if dtype.hasobject or path.stat().st_size - stream.tell() != math.prod(shape) * dtype.itemsize:
            raise ValueError("invalid checkpoint array length/dtype")
        digest = hashlib.sha256(repr((dtype.str, shape, bool(fortran))).encode())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _dtype_code(dtype):
    # dtype.str normalizes C long-long to C long on LP64 platforms. Preserve
    # the scalar type selected by the original dtype, including byte order.
    if np.dtype(dtype.str).type is not dtype.type:
        return dtype.byteorder + dtype.char if dtype.byteorder in "<>=" else dtype.char
    return dtype.str


class _Values:
    """Only the concrete field/container types used by preparation, with aliases."""
    def __init__(self, store=None, identity=False):
        self.store, self.identity = store, identity
        self.nodes, self.buffers, self.seen, self.owners, self.active = [], [], {}, {}, set()
        self.matrix_owners = set()

    def pack(self, value):
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is float:
            return ["float", struct.pack("!d", value).hex()]
        if isinstance(value, np.generic):
            if type(value) is not value.dtype.type:
                raise UnsupportedState("custom scalar type")
            if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.metadata:
                raise UnsupportedState("object/structured scalars")
            return ["scalar", _dtype_code(value.dtype), value.tobytes().hex()]
        if isinstance(value, Path):
            if type(value) is not type(Path()):
                raise UnsupportedState("custom path type")
            if self.identity:
                return ["path", str(value), _path_content(value)]
            if self.store and (value.resolve() == self.store.scratch or self.store.scratch in value.resolve().parents):
                raise UnsupportedState(f"state depends on caller scratch path: {value}")
            return ["path", str(value)]
        matrix_fields = ("similarity", "distance", "cluster_similarity", "cluster_distance")
        candidates = (getattr(value, name) for name in matrix_fields) if isinstance(value, DataSet) else (
            (value.get(name) for name in matrix_fields) if isinstance(value, dict) else ())
        for array in candidates:
            if isinstance(array, np.ndarray):
                while isinstance(array.base, np.ndarray):
                    array = array.base
                self.matrix_owners.add(id(array))
        key = id(value)
        if key in self.active:
            raise UnsupportedState("cyclic preparation state")
        if key in self.seen:
            return ["ref", self.seen[key]]
        index = len(self.nodes)
        self.seen[key] = index
        self.nodes.append(None)
        self.active.add(key)
        if isinstance(value, np.ndarray):
            if type(value) not in (np.ndarray, np.memmap):
                raise UnsupportedState("custom array type")
            if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.metadata:
                raise UnsupportedState("object/structured arrays")
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if type(owner) not in (np.ndarray, np.memmap) or owner.dtype.metadata:
                raise UnsupportedState("custom array storage")
            if not (owner.flags.c_contiguous or owner.flags.f_contiguous):
                raise UnsupportedState("noncontiguous root storage")
            owner_id = id(owner)
            if owner_id not in self.owners:
                self.owners[owner_id] = len(self.buffers)
                record = {"shape": list(owner.shape), "dtype": owner.dtype.str,
                          "fortran": bool(owner.flags.f_contiguous)}
                if self.identity:
                    record["content"] = _array_key(owner)
                elif owner.nbytes < 65536 and owner_id not in self.matrix_owners:
                    record["inline"] = (memoryview(owner.ravel(order="K")).cast("B").hex() if owner.nbytes else "")
                else:
                    record.update(self.store.array(owner))
                self.buffers.append(record)
            node = ["array", self.owners[owner_id], list(value.shape), _dtype_code(value.dtype), list(value.strides),
                    value.ctypes.data - owner.ctypes.data, bool(value.flags.writeable), isinstance(value, np.memmap)]
        elif type(value) is DataSet:
            if self.store and value.type == "P" and value.format == "PDB":
                for path in (value.data or {}).values():
                    resolved = Path(path).resolve()
                    if resolved == self.store.scratch or self.store.scratch in resolved.parents:
                        raise UnsupportedState(f"protein state depends on caller scratch: {path}")
            schema = [f.name for f in fields(DataSet)]
            if set(vars(value)) != set(schema):
                raise UnsupportedState("unexpected DataSet fields")
            node = ["dataset", [[name, self.pack(getattr(value, name))] for name in schema]]
        elif type(value) is Namespace:
            node = ["namespace", self.pack(vars(value))]
        elif type(value) in (dict, defaultdict):
            factory = None
            if isinstance(value, defaultdict):
                fn = value.default_factory
                if fn in (None, float, int):
                    factory = None if fn is None else fn.__name__
                elif getattr(fn, "__code__", None) and fn.__module__ in ("datasail.cluster.clustering", "datasail.solver.overflow", __name__):
                    default = fn()
                    if not isinstance(default, np.ndarray) or np.any(default):
                        raise UnsupportedState("unknown default factory")
                    factory = ["zeros", default.dtype.str, list(default.shape)]
                else:
                    raise UnsupportedState("unknown default factory")
            node = ["defaultdict" if isinstance(value, defaultdict) else "dict", factory,
                    [[self.pack(k), self.pack(v)] for k, v in value.items()]]
        elif type(value) in (list, tuple, set, frozenset):
            items = list(value)
            if type(value) in (set, frozenset):
                items.sort(key=lambda x: (type(x).__name__, repr(x)))
            node = [type(value).__name__, [self.pack(v) for v in items]]
        else:
            raise UnsupportedState(f"unsupported preparation value: {type(value).__name__}")
        self.nodes[index] = node
        self.active.remove(key)
        return ["ref", index]

    def document(self, value):
        root = self.pack(value)
        return {"root": root, "nodes": self.nodes, "buffers": self.buffers}


def _unpack(document, directory):
    nodes, buffers, loaded, storage, checked = document["nodes"], document["buffers"], {}, {}, set()

    def get(value):
        if value is None or type(value) in (str, int, bool):
            return value
        tag = value[0]
        if tag == "float":
            return struct.unpack("!d", bytes.fromhex(value[1]))[0]
        if tag == "scalar":
            return np.frombuffer(bytes.fromhex(value[2]), dtype=value[1])[0]
        if tag == "path":
            return Path(value[1])
        if tag != "ref":
            raise ValueError("unknown preparation value tag")
        index = value[1]
        if index in loaded:
            return loaded[index]
        node = nodes[index]
        kind = node[0]
        if kind == "array":
            number = node[1]
            if number not in storage:
                record = buffers[number]
                if "inline" in record:
                    base = np.ndarray(record["shape"], dtype=record["dtype"],
                                      buffer=bytearray.fromhex(record["inline"]), order="F" if record["fortran"] else "C")
                else:
                    filename = record["file"]
                    if re.fullmatch(r"[0-9a-f]{64}\.npy", filename) is None:
                        raise ValueError("invalid preparation array filename")
                    path = directory / filename
                    if filename not in checked:
                        if _sha(path) != record["sha256"]:
                            raise ValueError("preparation array checksum differs")
                        checked.add(filename)
                    base = np.load(path, mmap_mode="c", allow_pickle=False)
                    if list(base.shape) != record["shape"] or base.dtype.str != record["dtype"]:
                        raise ValueError("preparation array shape/dtype differs")
                storage[number] = base
            result = np.ndarray(node[2], dtype=node[3], buffer=storage[number], offset=node[5], strides=node[4])
            if node[7]:
                result = result.view(np.memmap)
            result.flags.writeable = node[6]
        elif kind == "dataset":
            values = {name: get(v) for name, v in node[1]}
            if set(values) != {f.name for f in fields(DataSet)}:
                raise ValueError("DataSet schema differs")
            result = DataSet(**values)
        elif kind == "namespace":
            result = Namespace(**get(node[1]))
        elif kind in ("dict", "defaultdict"):
            factory = node[1]
            if isinstance(factory, list) and factory[0] == "zeros":
                dtype, shape = factory[1:]
                def factory():
                    return np.zeros(shape, dtype=dtype)
            else:
                factory = {None: None, "float": float, "int": int}[factory]
            result = {} if kind == "dict" else defaultdict(factory)
            result.update((get(k), get(v)) for k, v in node[2])
        elif kind in ("list", "tuple", "set", "frozenset"):
            result = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](get(v) for v in node[1])
        else:
            raise ValueError("unknown preparation node")
        loaded[index] = result
        return result
    return get(document["root"])


def _path_content(path):
    if path.is_file():
        return _sha(path)
    if path.is_dir():
        if any(not p.is_file() for p in path.iterdir()):
            raise UnsupportedState("nested source directories")
        return [[p.name, _sha(p)] for p in path.iterdir()]
    return None


def _signature(value):
    return _fingerprint(_Values(identity=True).document(value))


def _implementation(kind, methods=()):
    root = Path(__file__).resolve().parent.parent
    paths = [*root.joinpath("reader").glob("*.py"), root / "parsers.py", root / "settings.py"]
    paths += [root / "cluster" / name for name in (
        ["ecfp.py", "vectors.py"] if kind == "ligand" else [p.name for p in root.joinpath("cluster").glob("*.py")])]
    if kind == "preparation":
        paths += [root / "routine.py", root / "solver/overflow.py", root / "solver/cluster_2d.py", root / "argparse_patch.py"]
    packages = ["numpy", "scipy", "scikit-learn", "pandas", "pyarrow", "rdkit", "h5py"]
    tools = {}
    binaries = {"foldseek": "foldseek", "cdhit": "cd-hit", "cdhit_est": "cd-hit-est",
                "mmseqs": "mmseqs", "mmseqs2": "mmseqs", "mmseqspp": "mmseqs", "diamond": "diamond", "mash": "mash"}
    for method in methods:
        for name in (binaries if method is None else [method] if isinstance(method, str) else []):
            if name in binaries:
                path = shutil.which(binaries[name])
                tools[name] = None if path is None else _sha(Path(path))
    return {"source": {str(p.relative_to(root)): _sha(p) for p in sorted(set(paths))},
            "dependencies": {name: importlib.metadata.version(name) for name in packages},
            "tools": tools}


class _Store:
    def __init__(self, directory, scratch):
        self.directory, self.scratch = directory, Path(scratch or "tmp").resolve()
        self.source_ids = {}

    def array(self, array):
        filename = _array_key(array) + ".npy"
        path = self.directory / filename
        # Existing blobs are immutable. Never deduplicate through mutable mappings.
        valid = False
        if path.exists():
            try:
                valid = _file_array_key(path) + ".npy" == filename
            except (OSError, ValueError):
                pass
        if not valid and not path.exists():
            for axis in ("e", "f"):
                native = self.directory.parent / axis / "similarity.npy"
                marker = native.parent / "matrix.json"
                owner = native.parent / "owner.json"
                if native.is_file() and marker.is_file() and owner.is_file():
                    try:
                        with _locked(native.parent, _read_json(owner)["search"]):
                            if _sha(native) == _read_json(marker)["sha256"] and _file_array_key(native) + ".npy" == filename:
                                os.link(native, path)
                                _sync_directory(self.directory)
                                valid = True
                                break
                    except (OSError, ValueError, KeyError, RuntimeError):
                        pass  # Different filesystems or invalid native checkpoint: use ordinary publication.
        if not valid:
            partial = self.directory / "array.part"
            with partial.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, path)
            _sync_directory(self.directory)
        return {"file": filename, "sha256": _sha(path)}

    def load(self, stage, identity):
        path = self.directory / (stage + ".json")
        if not path.exists():
            return None
        try:
            record = _read_json(path)
            if (record["format"], record["version"], record["schema"], record["identity"]) != (_FORMAT, _VERSION, _SCHEMA, identity):
                raise ValueError("stage input/settings/implementation identity differs")
            if _fingerprint(record["values"]) != record["state_sha256"]:
                raise ValueError("preparation metadata checksum differs")
            result = _unpack(record["values"], self.directory)
            LOGGER.info("Reused completed %s checkpoint: %s", stage, path)
            return result
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            LOGGER.warning("Cannot reuse %s checkpoint (%s); trying earlier stages", stage, error)
            return None

    def save(self, stage, identity, value):
        try:
            values = _Values(self).document(value)
        except UnsupportedState as error:
            LOGGER.warning("Did not checkpoint %s: %s", stage, error)
            return
        _unpack(values, self.directory)  # Validate the entire graph and all artifacts before completion.
        _publish_json(self.directory / (stage + ".json"), {
            "format": _FORMAT, "version": _VERSION, "schema": _SCHEMA, "identity": identity, "state_sha256": _fingerprint(values), "values": values})
        self.collect()
        LOGGER.info("Published completed %s checkpoint: %s", stage, self.directory)

    def collect(self):
        referenced = set()
        try:
            for stage in _STAGES:
                path = self.directory / (stage + ".json")
                if path.exists():
                    record = _read_json(path)
                    if _fingerprint(record["values"]) != record["state_sha256"]:
                        return
                    referenced.update(b["file"] for b in record["values"]["buffers"] if "file" in b)
        except (OSError, ValueError, KeyError, TypeError):
            return  # Preserve artifacts whose corrupt completion record needs diagnosis.
        for path in self.directory.glob("*.npy"):
            if path.name not in referenced:
                path.unlink()


@contextmanager
def _using(directory, scratch):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Preparation checkpoint directory must not be a symlink")
    directory = directory.resolve()
    current = _ACTIVE.get()
    if current is not None and current.directory == directory:
        yield current
        return
    with _locked(directory, {"kind": "preparation-state", "schema": _SCHEMA}, allowed=_allowed):
        store = _Store(directory, scratch)
        token = _ACTIVE.set(store)
        try:
            yield store
        finally:
            _ACTIVE.reset(token)


def _rng():
    return random.getstate(), np.random.get_state()


def _restore_rng(states):
    random.setstate(states[0])
    np.random.set_state(states[1])


def _dataset_identity(dataset):
    from datasail.cluster._foldseek_checkpoint import _inputs
    contents = _inputs(dataset) if dataset.type == "P" and (dataset.format == "PDB" or
                isinstance(dataset.similarity, str) and dataset.similarity.lower() == "foldseek") else None
    return _signature((dataset, contents))


def ligand(dataset, directory, compute):
    stage = "ligand-" + Path(directory).name
    if stage not in _STAGES:
        return compute(dataset)
    identity = _signature((dataset.type, dataset.format, dataset.names, dataset.data, dataset.id_map, _rng(),
                           _implementation("ligand")))
    with _using(Path(directory).parent / "state", None) as store:
        state = store.load(stage, identity)
        changed = ("names", "data", "id_map", "cluster_names", "cluster_map", "cluster_similarity", "cluster_distance")
        if state is not None:
            values, rng = state
            for name, value in values.items():
                setattr(dataset, name, value)
            _restore_rng(rng)
            return
        compute(dataset)
        store.save(stage, identity, ({name: getattr(dataset, name) for name in changed}, _rng()))


def _coverage(dataset, original_ids):
    accepted = list(dataset.id_map or {})
    return {"input_ids": original_ids, "accepted_ids": accepted,
            "id_map": dataset.id_map,
            "excluded_ids": None if original_ids is None else [n for n in original_ids if n not in (dataset.id_map or {})]}


def clustered(dataset, kwargs, compute):
    directory = Path(kwargs["foldseek_checkpoint_dir"])
    stage = "cluster-" + directory.name
    try:
        identity = _signature((_dataset_identity(dataset), kwargs["linkage"], kwargs["threads"], _rng(),
                               (kwargs.get("cache"), kwargs.get("cache_dir")) if kwargs.get("cache") else None,
                               _implementation("cluster", tuple(m for m in (dataset.similarity, dataset.distance) if m is not None))))
    except UnsupportedState as error:
        LOGGER.warning("Did not checkpoint clustering: %s", error)
        return compute(dataset, **kwargs)
    with _using(directory.parent / "state", kwargs.get("tmp_dir")) as store:
        state = store.load(stage, identity) if stage in _STAGES else None
        if state is not None:
            result, rng, coverage = state
            _restore_rng(rng)
            return result
        original_ids = store.source_ids.get(directory.name)
        if original_ids is None:
            # At a standalone clustering boundary only the reader's accepted
            # IDs are known. Capture them before similarity mutates the dataset.
            original_ids = list(dict.fromkeys([*(dataset.data or {}), *(dataset.id_map or {})]))
        result = compute(dataset, **kwargs)
        if stage in _STAGES:
            store.save(stage, identity, (result, _rng(), _coverage(result, original_ids)))
        return result


def _request_identity(kwargs):
    keys = [k for k in kwargs if k.startswith(("e_", "f_")) or k in
            ("inter", "inter_strat", "techniques", "splits", "names", "overflow", "linkage", "threads")]
    values = {k: kwargs[k] for k in keys}
    if kwargs.get("cache"):
        values["legacy_cache"] = (True, kwargs.get("cache_dir"))
    # Dictionary values used as protein filenames are strings in some callers.
    for axis in ("e", "f"):
        if values.get(axis + "_type") == "P":
            source = values.get(axis + "_data")
            if isinstance(source, dict):
                metric = values.get(axis + "_sim")
                structures = isinstance(metric, str) and metric.lower() == "foldseek"
                values[axis + "_data"] = {k: Path(v) if isinstance(v, str) and (structures or v.lower().endswith(".pdb")) else v for k, v in source.items()}
            elif isinstance(source, Path) and source.suffix in (".csv", ".tsv"):
                from datasail.reader.utils import read_csv
                values[axis + "_files"] = [Path(v) for _, v in read_csv(source, "," if source.suffix == ".csv" else "\t")]
            elif isinstance(source, Path) and source.suffix in (".pkl", ".h5"):
                raise UnsupportedState("indirect protein paths in binary input")
    methods = [kwargs.get(a + "_" + kind) for a in ("e", "f") if kwargs.get(a + "_type") is not None
               for kind in ("sim", "dist") if kwargs.get(a + "_" + kind) is not None]
    if any(kwargs.get(a + "_type") in ("P", "G") and kwargs.get(a + "_sim") is None and kwargs.get(a + "_dist") is None for a in ("e", "f")):
        methods.append(None)
    return _signature((values, _implementation("preparation", methods)))


def preparation(kwargs, compute):
    try:
        identity = _request_identity(kwargs)
    except UnsupportedState as error:
        LOGGER.warning("Did not checkpoint complete preparation: %s", error)
        return compute(kwargs)
    with _using(Path(kwargs["foldseek_checkpoint_dir"]) / "state", kwargs.get("tmp_dir")) as store:
        state = store.load("preparation", identity)
        if state is not None:
            result, rng, coverage = state
            _restore_rng(rng)
            return result
        store.source_ids = {axis: list(kwargs[axis + "_data"]) if isinstance(kwargs.get(axis + "_data"), dict) else None
                            for axis in ("e", "f")}
        result = compute(kwargs)
        coverage = {axis: _coverage(dataset, store.source_ids[axis]) for axis, dataset in zip(("e", "f"), result[:2])}
        store.save("preparation", identity, (result, _rng(), coverage))
        return result
