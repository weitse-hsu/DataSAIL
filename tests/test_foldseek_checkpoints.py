"""Completed Foldseek phases survive failure without changing matrix bytes."""
import hashlib
import json
import multiprocessing
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from datasail.cluster import _foldseek_checkpoint as checkpoint
from datasail.cluster import foldseek
from datasail.reader.read_proteins import read_protein_data
from tests.foldseek_helpers import assert_bits

ROWS = ("P.pdb_A\tQ.pdb_X\t0.8\t100\t0.9\n"
        "P.pdb_B\tQ.pdb_Y\t0.2\t50\t0.8\n"
        "Q.pdb_X\tP.pdb_A\t0.5\t80\t0.9\n"
        "Q.pdb_Y\tP.pdb_B\t0.1\t20\t0.8\n")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ["P", "Q"]:
        (inputs / (name + ".pdb")).write_text("END\n")
    monkeypatch.setitem(foldseek.INSTALLED, foldseek.FOLDSEEK, True)
    monkeypatch.setattr(checkpoint, "_tool", lambda name: {"version": name + " test", "sha256": "abc"})
    counts = {"search": 0, "matrix": 0}

    def search(command):
        counts["search"] += 1
        words = shlex.split(command)
        results = Path(words[words.index("mkdir") + 1])
        results.mkdir()
        (results / "aln.m8").write_text(ROWS)
        return 0

    original = foldseek._read_foldseek_similarity

    def matrix(*args, **kwargs):
        counts["matrix"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(foldseek.os, "system", search)
    monkeypatch.setattr(foldseek.subprocess, "call", lambda command, **kwargs: search(command))
    monkeypatch.setattr(foldseek, "_read_foldseek_similarity", matrix)

    def run(checkpoints=True, scratch="scratch", order=None):
        data = read_protein_data(inputs, sim="foldseek", detect_duplicates=False)
        if order is not None:
            data.names = list(order)
        temporary = tmp_path / scratch
        temporary.mkdir(exist_ok=True)
        foldseek.run_foldseek(data, tmp_dir=temporary,
                              checkpoint_dir=tmp_path / "checkpoints" if checkpoints else None)
        return data
    return run, counts, inputs


def test_disabled_fresh_and_resumed_exact_bytes(tmp_path, setup):
    run, counts, _ = setup
    disabled = run(False)
    fresh = run()
    resumed = run(scratch="another-scratch")
    assert counts == {"search": 2, "matrix": 2}
    assert disabled.cluster_names == fresh.cluster_names == resumed.cluster_names
    assert disabled.cluster_map == fresh.cluster_map == resumed.cluster_map
    assert_bits(disabled.cluster_similarity, fresh.cluster_similarity)
    assert_bits(fresh.cluster_similarity, resumed.cluster_similarity)
    # Independently selected chains: P->Q=.6, Q->P=.42, symmetric mean=.51.
    assert_bits(fresh.cluster_similarity, np.array([[1., .51], [.51, 1.]]))
    assert isinstance(resumed.cluster_similarity, np.memmap)
    assert resumed.cluster_similarity.mode == "c"
    resumed.cluster_similarity[:] = 99
    assert_bits(run().cluster_similarity, fresh.cluster_similarity)
    assert set(p.name for p in (tmp_path / "checkpoints").iterdir()) == {
        "owner.json", ".lock", "alignment.m8", "alignment.json", "similarity.npy", "matrix.json"}
    assert not list(tmp_path.glob("checkpoints/**/chains*"))


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failure_after_alignment_preserves_search(tmp_path, setup, monkeypatch, error):
    run, counts, _ = setup
    original = foldseek._read_foldseek_similarity
    monkeypatch.setattr(foldseek, "_read_foldseek_similarity", lambda *a, **kw: (_ for _ in ()).throw(error("interrupted")))
    with pytest.raises(error, match="interrupted"):
        run()
    assert (tmp_path / "checkpoints/alignment.json").is_file()
    assert not (tmp_path / "checkpoints/matrix.json").exists()
    monkeypatch.setattr(foldseek, "_read_foldseek_similarity", original)
    assert_bits(run(scratch="relaunched").cluster_similarity, np.array([[1., .51], [.51, 1.]]))
    assert counts == {"search": 1, "matrix": 1}


@pytest.mark.parametrize("phase", ["array", "completion"])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_interrupted_matrix_publication_recovers_alignment(tmp_path, setup, monkeypatch, phase, error):
    run, counts, _ = setup
    function = np.save if phase == "array" else checkpoint._publish_json

    def fail(*args, **kwargs):
        if phase == "array":
            args[0].write(b"partial array")
            raise error("disk interrupted")
        if args[0].name == "matrix.json":
            raise error("disk interrupted")
        return function(*args, **kwargs)

    target, name = (np, "save") if phase == "array" else (checkpoint, "_publish_json")
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(error, match="disk interrupted"):
        run()
    assert (tmp_path / "checkpoints/alignment.json").is_file()
    assert not (tmp_path / "checkpoints/matrix.json").exists()
    monkeypatch.setattr(target, name, function)
    assert_bits(run(scratch="after-write-failure").cluster_similarity, np.array([[1., .51], [.51, 1.]]))
    assert counts == {"search": 1, "matrix": 2}


@pytest.mark.parametrize("filename", ["similarity.npy", "matrix.json"])
def test_corrupt_matrix_recovers_from_alignment(tmp_path, setup, filename):
    run, counts, _ = setup
    expected = run().cluster_similarity
    path = tmp_path / "checkpoints" / filename
    path.write_bytes(path.read_bytes()[:7])
    assert_bits(run(scratch="restart").cluster_similarity, expected)
    assert counts == {"search": 1, "matrix": 2}


def test_corrupt_alignment_fails_without_overwriting(tmp_path, setup):
    run, counts, _ = setup
    run()
    (tmp_path / "checkpoints/matrix.json").unlink()
    path = tmp_path / "checkpoints/alignment.m8"
    path.write_text("truncated")
    with pytest.raises(ValueError, match="Invalid completed Foldseek alignment"):
        run()
    assert path.read_text() == "truncated"
    assert counts == {"search": 1, "matrix": 1}


@pytest.mark.parametrize("change", ["contents", "order", "filename", "arguments", "threads", "tool"])
def test_search_identity_changes_are_rejected(tmp_path, setup, monkeypatch, change):
    run, counts, inputs = setup
    run()
    data = read_protein_data(inputs, sim="foldseek", detect_duplicates=False)
    kwargs = {}
    if change == "contents":
        (inputs / "P.pdb").write_text("ATOM changed\n")
    elif change == "order":
        data.names.reverse()
    elif change == "filename":
        copied = inputs / "different.pdb"
        shutil.copyfile(inputs / "P.pdb", copied)
        data.data["P"] = copied
    elif change == "arguments":
        data.args.alignment_type = 1
    elif change == "threads":
        kwargs["threads"] = 2
    else:
        monkeypatch.setattr(checkpoint, "_tool", lambda name: {"version": "different", "sha256": "def"})
    before = (tmp_path / "checkpoints/owner.json").read_bytes()
    with pytest.raises(ValueError, match="Incompatible Foldseek checkpoint"):
        foldseek.run_foldseek(data, tmp_dir=tmp_path / "scratch", checkpoint_dir=tmp_path / "checkpoints", **kwargs)
    assert (tmp_path / "checkpoints/owner.json").read_bytes() == before
    assert counts == {"search": 1, "matrix": 1}


def test_matrix_implementation_change_reuses_search(tmp_path, setup, monkeypatch):
    run, counts, _ = setup
    expected = run().cluster_similarity
    version = checkpoint.np.__version__
    monkeypatch.setattr(checkpoint.np, "__version__", version + "-different")
    assert_bits(run().cluster_similarity, expected)
    assert counts == {"search": 1, "matrix": 2}


def test_unowned_files_and_unsafe_paths_are_preserved(tmp_path, setup):
    run, _, inputs = setup
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    owned = directory / "similarity.npy"
    owned.write_text("caller")
    with pytest.raises(ValueError, match="unowned"):
        run()
    assert owned.read_text() == "caller"
    for checkpoints, temporary in [(tmp_path, tmp_path / "scratch"), (tmp_path / "scratch/checkpoints", tmp_path / "scratch")]:
        with pytest.raises(ValueError, match="separate from tmp_dir"):
            foldseek.run_foldseek(read_protein_data(inputs, sim="foldseek"), tmp_dir=temporary,
                                  checkpoint_dir=checkpoints)


def test_public_api_reuses_after_downstream_failure_and_cleans_only_scratch(tmp_path, setup, monkeypatch):
    from datasail.sail import datasail
    import datasail.routine as routine
    _, counts, inputs = setup
    original = routine.check_dataset
    monkeypatch.setattr(routine, "check_dataset", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("downstream")))
    arguments = dict(techniques=["C1f", "I1f"], f_type="P", f_data=inputs, f_sim="foldseek", f_args="X",
                     splits=[1, 1], names=["train", "test"], epsilon=0.5, delta=0.5,
                     threads=1, max_sec=2, foldseek_checkpoint_dir=tmp_path / "persistent")
    with pytest.raises(RuntimeError, match="downstream"):
        datasail(**arguments, tmp_dir=tmp_path / "first-api-scratch")
    assert not (tmp_path / "first-api-scratch").exists()
    assert (tmp_path / "persistent/f/matrix.json").is_file()
    monkeypatch.setattr(routine, "check_dataset", original)
    result = datasail(**arguments, tmp_dir=tmp_path / "second-api-scratch")
    assert counts == {"search": 1, "matrix": 1}
    assert set(result[1]) == {"C1f", "I1f"}
    assert not (tmp_path / "second-api-scratch").exists()


def hold_lock(path, identity, ready, release):
    with checkpoint._locked(Path(path), identity):
        ready.set()
        release.wait(20)


def test_concurrent_writer_is_rejected_and_lock_survives_process_death(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    path = tmp_path / "persistent"
    process = context.Process(target=hold_lock, args=(str(path), {"input": "x"}, ready, release))
    process.start()
    try:
        assert ready.wait(20)
        with pytest.raises(RuntimeError, match="in use"):
            with checkpoint._locked(path, {"input": "x"}):
                pytest.fail("two writers")
        process.terminate()
        process.join(20)
        with checkpoint._locked(path, {"input": "x"}):
            assert (path / "owner.json").is_file()
    finally:
        if process.is_alive():
            process.terminate()
        process.join()


@pytest.mark.parametrize("rows", ["P.pdb\tQ.pdb\t-0.0\t100\t1\n", "P.pdb\tQ.pdb\t-NAN\t100\t1\nQ.pdb\tP.pdb\tNAN\t100\t1\n"])
def test_checkpoint_preserves_supported_special_value_bits(tmp_path, setup, rows, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "ROWS", rows)
    run, _, _ = setup
    expected = run(False).cluster_similarity
    assert_bits(run().cluster_similarity, expected)
    assert_bits(run().cluster_similarity, expected)


@pytest.mark.parametrize("change", ["shape", "dtype", "names"])
def test_invalid_matrix_contract_recovers_alignment(tmp_path, setup, change):
    run, counts, _ = setup
    expected = run().cluster_similarity
    directory = tmp_path / "checkpoints"
    record = json.loads((directory / "matrix.json").read_text())
    if change == "names":
        record["names"].reverse()
    else:
        array = np.ones((1, 1)) if change == "shape" else expected.astype(np.float32)
        np.save(directory / "similarity.npy", array)
        record["sha256"] = hashlib.sha256((directory / "similarity.npy").read_bytes()).hexdigest()
    (directory / "matrix.json").write_text(json.dumps(record))
    assert_bits(run().cluster_similarity, expected)
    assert counts == {"search": 1, "matrix": 2}


def test_failed_search_and_source_change_are_not_published(tmp_path, setup, monkeypatch):
    run, _, inputs = setup
    monkeypatch.setattr(foldseek.os, "system", lambda command: 1)
    monkeypatch.setattr(foldseek.subprocess, "call", lambda command, **kwargs: 1)
    with pytest.raises(ValueError, match="command failed"):
        run()
    assert not (tmp_path / "checkpoints/alignment.json").exists()
    assert not (tmp_path / "checkpoints/.search").exists()
    original = foldseek.shutil.copy

    def changed(source, target):
        result = original(source, target)
        (Path(target) / Path(source).name).write_text("changed during staging")
        return result

    monkeypatch.setattr(foldseek.shutil, "copy", changed)
    with pytest.raises(ValueError, match="changed during staging"):
        run()
    assert not (tmp_path / "checkpoints/alignment.json").exists()
    assert (inputs / "P.pdb").read_text() == "END\n"


def test_public_api_rejects_overlap_before_cleanup(tmp_path):
    from datasail.sail import datasail
    protected = tmp_path / "persistent"
    protected.mkdir()
    (protected / "sentinel").write_text("keep")
    with pytest.raises(ValueError, match="separate from tmp_dir"):
        datasail(foldseek_checkpoint_dir=protected, tmp_dir=protected / "scratch")
    assert (protected / "sentinel").read_text() == "keep"
    assert not (protected / "scratch").exists()


def test_owned_artifact_symlink_is_rejected(tmp_path, setup):
    run, _, inputs = setup
    run()
    target = tmp_path / "checkpoints/similarity.npy"
    target.unlink()
    target.symlink_to(inputs / "P.pdb")
    with pytest.raises(ValueError, match="symlinks"):
        run()
    assert (inputs / "P.pdb").read_text() == "END\n"


def test_both_foldseek_axes_and_combined_techniques(tmp_path, setup):
    from datasail.sail import datasail
    _, counts, inputs = setup
    arguments = dict(techniques=["C2", "C1e", "C1f"], e_type="P", f_type="P",
                     e_data=inputs, f_data=inputs, e_sim="foldseek", f_sim="foldseek",
                     inter=[("P", "P"), ("Q", "Q")], e_args="X", f_args="X",
                     splits=[1, 1], names=["train", "test"], epsilon=0.5, delta=0.5,
                     threads=1, max_sec=2, foldseek_checkpoint_dir=tmp_path / "persistent")
    fresh = datasail(**arguments, tmp_dir=tmp_path / "first")
    resumed = datasail(**arguments, tmp_dir=tmp_path / "second")
    assert counts == {"search": 2, "matrix": 2}
    assert fresh == resumed
    assert (tmp_path / "persistent/e/matrix.json").is_file()
    assert (tmp_path / "persistent/f/matrix.json").is_file()


@pytest.mark.skipif(not shutil.which("foldseek"), reason="native Foldseek unavailable")
def test_native_search_and_fresh_process_recovery(tmp_path, monkeypatch):
    inputs = tmp_path / "native inputs"
    inputs.mkdir()
    fixtures = Path(__file__).parent / "raw_data/cyclophilins"
    first = (fixtures / "8cu5_a.pdb").read_text()
    # A real single chain plus a two-chain structure with HiQBind-style names.
    atoms = [line for line in first.splitlines() if line.startswith("ATOM")]
    multi = "\n".join(atoms + [line[:21] + "B" + line[22:] for line in atoms]) + "\nEND\n"
    (inputs / "5abc_LIG_A_101.pdb").write_text(first)
    (inputs / "6def_LIG_B_202.pdb").write_text(multi)
    calls = {"search": 0, "matrix": 0}
    search, build = foldseek._run_foldseek_search, foldseek._read_foldseek_similarity

    def counted_search(*args, **kwargs):
        calls["search"] += 1
        return search(*args, **kwargs)

    def counted_build(*args, **kwargs):
        calls["matrix"] += 1
        return build(*args, **kwargs)

    monkeypatch.setattr(foldseek, "_run_foldseek_search", counted_search)
    monkeypatch.setattr(foldseek, "_read_foldseek_similarity", counted_build)
    # Source identity must describe the real implementation, not instrumentation.
    original_source = checkpoint.inspect.getsource
    monkeypatch.setattr(checkpoint.inspect, "getsource",
                        lambda obj: original_source(search if obj is counted_search else obj))
    matrices = []
    for mode in ["disabled", "fresh", "resumed"]:
        scratch = tmp_path / mode
        scratch.mkdir()
        data = read_protein_data(inputs, sim="foldseek", detect_duplicates=False)
        foldseek.run_foldseek(data, threads=1, tmp_dir=scratch,
                              checkpoint_dir=None if mode == "disabled" else tmp_path / "persistent")
        matrices.append(data.cluster_similarity)
    assert calls == {"search": 2, "matrix": 2}
    assert_bits(matrices[0], matrices[1])
    assert_bits(matrices[1], matrices[2])
    script = tmp_path / "restart.py"
    script.write_text("""
import json, sys
from pathlib import Path
import numpy as np
import datasail.cluster.foldseek as foldseek
from datasail.reader.read_proteins import read_protein_data
root = Path(sys.argv[1])
assert Path(foldseek.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
def forbidden(*args, **kwargs):
    raise AssertionError('completed phase reran in fresh process')
foldseek._read_foldseek_similarity = forbidden if sys.argv[3] == 'matrix' else foldseek._read_foldseek_similarity
data = read_protein_data(root / 'native inputs', sim='foldseek', detect_duplicates=False)
scratch = root / ('new-process-' + sys.argv[3])
scratch.mkdir()
# Native search cannot execute, but the real tool identity still verifies.
foldseek.os.system = forbidden
foldseek.subprocess.call = forbidden
foldseek.run_foldseek(data, threads=1, tmp_dir=scratch, checkpoint_dir=root / 'persistent')
np.save(root / ('result-' + sys.argv[3] + '.npy'), data.cluster_similarity)
(root / ('names-' + sys.argv[3] + '.json')).write_text(json.dumps(data.cluster_names))
""")
    repository = Path(foldseek.__file__).resolve().parents[2]
    for phase in ["matrix", "alignment"]:
        if phase == "alignment":
            (tmp_path / "persistent/matrix.json").unlink()
        subprocess.run([sys.executable, str(script), str(tmp_path), str(repository), phase],
                       env={**os.environ, "PYTHONPATH": str(repository)}, check=True, capture_output=True, text=True)
        assert_bits(np.load(tmp_path / ("result-" + phase + ".npy")), matrices[0])
        assert json.loads((tmp_path / ("names-" + phase + ".json")).read_text()) == data.cluster_names


def lock_with_child(path, ready):
    with checkpoint._locked(Path(path), {"input": "x"}) as fd:
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(3)"], pass_fds=(fd,))
        ready.set()
        child.wait()


def test_native_child_retains_lock_if_parent_dies(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    path = tmp_path / "persistent"
    process = context.Process(target=lock_with_child, args=(str(path), ready))
    process.start()
    try:
        assert ready.wait(20)
        process.terminate()
        process.join(20)
        with pytest.raises(RuntimeError, match="in use"):
            with checkpoint._locked(path, {"input": "x"}):
                pytest.fail("orphaned native command lost lock")
    finally:
        if process.is_alive():
            process.terminate()
        process.join()


@pytest.mark.parametrize("rows", ["", "P\tQ\t.2\t1\t.9", "P\tQ\t.2\t1\t.9\ninvalid\trow\n"])
def test_invalid_search_output_is_not_completed(tmp_path, setup, monkeypatch, rows):
    monkeypatch.setattr(sys.modules[__name__], "ROWS", rows)
    run, counts, _ = setup
    with pytest.raises(ValueError, match="Foldseek search"):
        run()
    assert not (tmp_path / "checkpoints/alignment.json").exists()
    assert not (tmp_path / "checkpoints/matrix.json").exists()
    assert counts == {"search": 1, "matrix": 0}
