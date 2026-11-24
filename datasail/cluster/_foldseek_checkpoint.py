"""Persistent recovery for the two expensive Foldseek phases, not clustering."""

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow

from datasail.settings import LOGGER

_FORMAT = "datasail-foldseek"
_VERSION = 1
_FILES = {"owner.json", ".lock", "alignment.m8", "alignment.json", "similarity.npy", "matrix.json",
          "owner.json.part", "alignment.json.part", "similarity.npy.part", "matrix.json.part", ".search"}


def validate_checkpoint_path(directory, tmp_dir):
    """Reject both containment directions before the public API creates scratch."""
    directory = Path(directory).expanduser()
    if directory.is_symlink():
        raise ValueError("Foldseek checkpoint directory must not be a symlink")
    directory = directory.resolve()
    temporary = Path(tmp_dir or "tmp").expanduser().resolve()
    if directory == temporary or directory in temporary.parents or temporary in directory.parents:
        raise ValueError("Foldseek checkpoint directory must be separate from tmp_dir (neither may contain the other)")
    return directory


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tool(name):
    executable = shutil.which(name)
    if executable is None:
        raise ValueError(f"Cannot verify Foldseek checkpoint compatibility: {name} executable is unavailable")
    path = Path(executable).resolve()
    argument = "version" if name == "foldseek" else "--version"
    version = subprocess.run([str(path), argument], check=True, capture_output=True, text=True).stdout.strip()
    return {"sha256": _sha(path), "version": version}


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_json(path, value):
    partial = path.with_name(path.name + ".part")
    with partial.open("w", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(partial, path)
    _sync_directory(path.parent)


def _read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


@contextmanager
def _locked(directory, identity):
    # Import lazily so disabled checkpointing and non-POSIX imports are unaffected.
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("Foldseek checkpoints require POSIX advisory file locks") from error
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Foldseek checkpoint directory must not be a symlink")
    entries = list(directory.iterdir())
    if not (directory / "owner.json").exists() and entries:
        raise ValueError(f"Foldseek checkpoint directory is unowned or being initialized; use an empty directory: {directory}")
    if any(p.name not in _FILES or p.is_symlink() for p in entries):
        raise ValueError(f"Foldseek checkpoint directory contains unrelated files or symlinks: {directory}")
    fd = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Foldseek checkpoint directory is in use: {directory}") from error
        owner = directory / "owner.json"
        expected = {"format": _FORMAT, "version": _VERSION, "search": identity}
        if owner.exists():
            try:
                recorded = _read_json(owner)
            except (OSError, ValueError) as error:
                raise ValueError(f"Invalid Foldseek checkpoint ownership metadata: {directory}") from error
            if recorded != expected:
                LOGGER.warning("Foldseek checkpoint search identity differs: %s", directory)
                raise ValueError(f"Incompatible Foldseek checkpoint inputs, filename mapping, search parameters or implementation/tool version; use a new directory: {directory}")
        else:
            _publish_json(owner, expected)
        yield fd
    finally:
        # Closing releases the kernel lock even after failure; process death does too.
        os.close(fd)


def _inputs(dataset):
    names = list(dataset.names)
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Foldseek checkpoints require nonempty unique ordered entity names")
    seen, records = {}, []
    for name in names:
        path = Path(dataset.data[name])
        resolved = path.resolve()
        if resolved not in seen:
            seen[resolved] = _sha(path)
        records.append({"name": name, "filename": path.name, "sha256": seen[resolved]})
    return records


def _matrix(directory, identity, names):
    metadata, artifact = directory / "matrix.json", directory / "similarity.npy"
    if not metadata.exists():
        LOGGER.info("No completed Foldseek matrix checkpoint: %s", directory)
        return None
    try:
        record = _read_json(metadata)
        if record["format"] != _FORMAT or record["version"] != _VERSION or record["processing"] != identity:
            raise ValueError("matrix-processing identity differs")
        if record["names"] != names:
            raise ValueError("matrix entity order differs")
        if _sha(artifact) != record["sha256"]:
            raise ValueError("matrix checksum differs")
        matrix = np.load(artifact, mmap_mode="c", allow_pickle=False)
        if matrix.dtype != np.dtype("float64") or matrix.shape != (len(names), len(names)):
            raise ValueError("matrix must be square native float64 in the requested entity order")
        LOGGER.info("Reused completed Foldseek similarity matrix: %s", artifact)
        return matrix
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOGGER.warning("Cannot reuse Foldseek matrix checkpoint (%s); trying alignment: %s", error, directory)
        return None


def _alignment(directory):
    metadata, artifact = directory / "alignment.json", directory / "alignment.m8"
    if not metadata.exists():
        LOGGER.info("No completed Foldseek alignment checkpoint: %s", directory)
        return None
    try:
        record = _read_json(metadata)
        if record["format"] != _FORMAT or record["version"] != _VERSION or _sha(artifact) != record["sha256"]:
            raise ValueError("alignment metadata or checksum differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOGGER.warning("Cannot reuse Foldseek alignment checkpoint: %s", error)
        raise ValueError(f"Invalid completed Foldseek alignment checkpoint; preserve it for diagnosis and use a new directory: {directory}") from error
    LOGGER.info("Reused completed Foldseek alignment: %s", artifact)
    return artifact


def _save_alignment(source, directory):
    # Validate physical records while computing the publication checksum. Keep
    # the ordinary reader's numeric inference and chain interpretation unchanged.
    if source.is_symlink() or not source.is_file():
        raise ValueError("Foldseek search produced no regular alignment")
    digest = hashlib.sha256()
    pending, records = b"", 0
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            rows = (pending + chunk).split(b"\n")
            pending = rows.pop()
            for row in rows:
                records += 1
                if row.count(b"\t") != 4:
                    raise ValueError(f"Foldseek search alignment row {records}: expected five tab-separated columns")
        if pending:
            raise ValueError("Foldseek search alignment ends in an incomplete record")
        if records == 0:
            raise ValueError("Foldseek search produced no nonempty regular alignment")
        os.fsync(stream.fileno())
    digest = digest.hexdigest()
    # Search runs on the checkpoint filesystem: rename, never copy the huge table.
    os.replace(source, directory / "alignment.m8")
    _sync_directory(directory)
    _publish_json(directory / "alignment.json", {"format": _FORMAT, "version": _VERSION, "sha256": digest})


def _save_matrix(matrix, directory, identity, names):
    if matrix.dtype != np.dtype("float64") or matrix.shape != (len(names), len(names)):
        raise ValueError("Cannot checkpoint a matrix with unexpected shape or dtype")
    partial = directory / "similarity.npy.part"
    with partial.open("wb") as target:
        np.save(target, matrix, allow_pickle=False)
        target.flush()
        os.fsync(target.fileno())
    digest = _sha(partial)
    os.replace(partial, directory / "similarity.npy")
    _sync_directory(directory)
    _publish_json(directory / "matrix.json", {"format": _FORMAT, "version": _VERSION,
                  "processing": identity, "names": names, "sha256": digest})


def checkpointed_foldseek(dataset, threads, log_dir, tmp_dir, checkpoint_dir):
    """Recover only search and matrix construction; later clustering is unchanged."""
    from datasail.cluster import foldseek
    from datasail.parsers import MultiYAMLParser
    from datasail.settings import FOLDSEEK

    directory = validate_checkpoint_path(checkpoint_dir, tmp_dir)
    scratch = Path(tmp_dir or "tmp").resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    records = _inputs(dataset)
    identity = {"entities": records, "tool": _tool("foldseek"), "threads": threads,
                "arguments": MultiYAMLParser(FOLDSEEK).get_user_arguments(dataset.args, []),
                "search_source": hashlib.sha256(inspect.getsource(foldseek._run_foldseek_search).encode()).hexdigest(),
                "parser_source": _sha(Path(inspect.getfile(MultiYAMLParser))),
                "argument_schema": _sha(Path(foldseek.__file__).parent.parent / "args/foldseek.yaml")}
    processing = {"source": _sha(Path(foldseek.__file__)), "numpy": np.__version__,
                  "pyarrow": pyarrow.__version__, "python": list(sys.version_info[:3]),
                  "byteorder": sys.byteorder, "sort": _tool("sort"), "search": _fingerprint(identity)}
    with _locked(directory, identity) as lock_fd:
        matrix = _matrix(directory, processing, list(dataset.names))
        if matrix is not None:
            return matrix
        alignment = _alignment(directory)
        if alignment is None:
            search = directory / ".search"
            if search.exists():
                shutil.rmtree(search)
            search.mkdir()
            expected = {r["filename"]: r["sha256"] for r in records}

            def verify_staged(staged):
                for filename, digest in expected.items():
                    if _sha(staged / filename) != digest:
                        raise ValueError(f"Foldseek input changed during staging: {filename}")

            try:
                alignment = foldseek._run_foldseek_search(dataset, threads, log_dir, scratch,
                                                          verify_staged=verify_staged, lock_fd=lock_fd,
                                                          results_folder=search / "fs_results")
                _save_alignment(alignment, directory)
            finally:
                shutil.rmtree(search, ignore_errors=True)
                shutil.rmtree(scratch / "fs_tmp", ignore_errors=True)
                shutil.rmtree(scratch / "fs_results", ignore_errors=True)
            alignment = directory / "alignment.m8"
            LOGGER.info("Published completed Foldseek alignment: %s", alignment)
        matrix = foldseek._read_foldseek_similarity(alignment, dataset, tmp_dir=tmp_dir,
                                                    threads=threads, sort_executable=foldseek._check_sort())
        _save_matrix(matrix, directory, processing, list(dataset.names))
        LOGGER.info("Published completed Foldseek similarity matrix: %s", directory / "similarity.npy")
        return matrix
