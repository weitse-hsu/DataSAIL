"""Exact preparation/model parity and recovery through the ordinary Python API."""
import copy
import importlib
import os
import pickle
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from datasail.cluster import _preparation_checkpoint as state
from datasail.reader.utils import DataSet
from datasail.sail import datasail
import datasail.routine as routine


def fixture():
    names = [f"L{i}" for i in range(8)]
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCN", "COC", "CN", "CCCl"]
    proteins = [f"P{i}" for i in range(8)]
    pairs = [(names[i], proteins[i]) for i in range(8)]
    pairs += [(names[i], proteins[(i + 1) % 8]) for i in range(8)]
    return dict(techniques=["R", "I1e", "C1f", "C2"], runs=2, splits=[1, 1], names=["train", "test"],
                e_type="M", e_data=dict(zip(names + ["duplicate", "invalid"], smiles + ["CC", "not_smiles"])),
                e_sim="ecfp", e_clusters=4, e_strat={n: {i % 2} for i, n in enumerate(names + ["duplicate", "invalid"])},
                f_type="O", f_data={n: n for n in proteins}, f_clusters=4,
                f_sim=(proteins, np.array([[float(i // 2 == j // 2) for j in range(8)] for i in range(8)])),
                f_strat={n: {i % 2} for i, n in enumerate(proteins)},
                inter=pairs + [("invalid", proteins[0]), ("duplicate", proteins[0])],
                inter_strat={p: {i % 2} for i, p in enumerate(pairs + [("invalid", proteins[0]), ("duplicate", proteins[0])])},
                delta=1., epsilon=1., threads=1, max_sec=5)


def canonical(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if sparse.issparse(value):
        value = value.tocsc()
        return (value.shape, canonical(value.data), canonical(value.indices), canonical(value.indptr))
    if isinstance(value, dict):
        return [(k, canonical(v)) for k, v in value.items()]
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, np.generic):
        return (value.dtype.str, value.tobytes())
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, (int, float, str, type(None))):
        return value
    return canonical(vars(value))


@pytest.fixture
def models(monkeypatch):
    observed = []
    from datasail.solver.utils import solve
    for module in ["id_1d", "id_2d", "cluster_1d", "cluster_2d"]:
        target = importlib.import_module("datasail.solver." + module)

        def inspect(loss, constraints, max_sec, solver, log_file, **kwargs):
            problem = cp.Problem(cp.Minimize(loss), constraints)
            data = problem.get_problem_data(cp.SCIP)[0]
            observed.append(([(v.shape, canonical(v.attributes)) for v in problem.variables()],
                             canonical({k: v for k, v in data.items() if k != "param_prob"})))
            return solve(loss, constraints, max_sec, solver, log_file, **kwargs)

        monkeypatch.setattr(target, "solve", inspect)
    return observed


def test_disabled_fresh_recovered_state_models_and_assignments(tmp_path, monkeypatch, models):
    baseline = datasail(**copy.deepcopy(fixture()), tmp_dir=tmp_path / "disabled")
    expected_models = copy.deepcopy(models)
    models.clear()
    observed = []
    original = routine._prepare

    def record(kwargs):
        result = original(kwargs)
        observed.append(state._signature(result))
        return result

    monkeypatch.setattr(routine, "_prepare", record)
    fresh = datasail(**copy.deepcopy(fixture()), tmp_dir=tmp_path / "fresh", checkpoint_dir=tmp_path / "checkpoint")
    assert fresh == baseline
    assert models == expected_models
    models.clear()
    monkeypatch.setattr(routine, "_prepare", lambda *a: pytest.fail("preparation reran"))
    original_solver = routine.run_solver

    def checked(**kwargs):
        assert "duplicate" in kwargs["e_dataset"].id_map
        assert "invalid" not in kwargs["e_dataset"].id_map
        return original_solver(**kwargs)

    monkeypatch.setattr(routine, "run_solver", checked)
    resumed = datasail(**copy.deepcopy(fixture()), tmp_dir=tmp_path / "different", checkpoint_dir=tmp_path / "checkpoint")
    assert resumed == baseline
    assert models == expected_models
    assert len(observed) == 1
    # Original coverage survives the reader's in-place input-dictionary edits.
    folder = tmp_path / "checkpoint/state"
    record = state._read_json(folder / "preparation.json")
    _, _, coverage = state._unpack(record["values"], folder)
    assert coverage["e"]["input_ids"] == list(fixture()["e_data"])
    assert coverage["e"]["excluded_ids"] == ["invalid"]
    assert coverage["e"]["id_map"]["duplicate"] == "L0"
    assert coverage["f"]["excluded_ids"] == []


@pytest.mark.parametrize("phase", ["ligand", "after-ligand", "after-e", "publication", "solver"])
def test_failure_boundaries_and_furthest_reuse(tmp_path, monkeypatch, phase):
    from datasail.cluster import clustering
    counts = {"ligand": 0, "cluster": 0, "overflow": 0}
    original_ligand, original_cluster, original_check = clustering.run_ecfp, clustering._cluster, routine.check_dataset

    def ligand(*args, **kwargs):
        counts["ligand"] += 1
        if phase == "ligand" and counts["ligand"] == 1:
            raise KeyboardInterrupt("during ligand")
        return original_ligand(*args, **kwargs)

    def cluster(ds, **kwargs):
        counts["cluster"] += 1
        if phase == "after-e" and ds.type == "O" and counts["cluster"] == 2:
            raise RuntimeError("after e")
        return original_cluster(ds, **kwargs)

    def check(*args, **kwargs):
        counts["overflow"] += 1
        return original_check(*args, **kwargs)

    monkeypatch.setattr(clustering, "run_ecfp", ligand)
    monkeypatch.setattr(clustering, "_cluster", cluster)
    monkeypatch.setattr(routine, "check_dataset", check)
    publish = state._publish_json
    failures = []

    def write(path, record):
        fail = (phase == "after-ligand" and path.name == "cluster-e.json") or (phase == "publication" and path.name == "preparation.json")
        if fail and not failures:
            failures.append(path)
            path.with_suffix(".json.part").write_text("partial")
            raise OSError("publication interrupted")
        return publish(path, record)

    monkeypatch.setattr(state, "_publish_json", write)
    solver = routine.run_solver
    if phase == "solver":
        monkeypatch.setattr(routine, "run_solver", lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt("solver interrupted")))
    arguments = dict(fixture(), checkpoint_dir=tmp_path / "checkpoint")
    with pytest.raises((KeyboardInterrupt, RuntimeError, OSError)):
        datasail(**copy.deepcopy(arguments), tmp_dir=tmp_path / "interrupted")
    before = counts.copy()
    monkeypatch.setattr(routine, "run_solver", solver)
    result = datasail(**copy.deepcopy(arguments), tmp_dir=tmp_path / "restarted")
    assert result[0] is not None
    assert counts["ligand"] == (2 if phase == "ligand" else 1)
    if phase == "solver":
        assert counts == before
    if phase in ("after-e", "publication"):
        assert counts["cluster"] == (3 if phase == "after-e" else 2)
    if phase == "after-ligand":
        assert before["ligand"] == counts["ligand"]


def test_corrupt_preparation_falls_back_to_axes(tmp_path, monkeypatch):
    args = dict(fixture(), checkpoint_dir=tmp_path / "checkpoint")
    expected = datasail(**copy.deepcopy(args), tmp_dir=tmp_path / "first")
    folder = tmp_path / "checkpoint/state"
    (folder / "preparation.json").write_text("truncated")
    from datasail.cluster import clustering
    monkeypatch.setattr(clustering, "_cluster", lambda *a, **kw: pytest.fail("valid axes recomputed"))
    assert datasail(**copy.deepcopy(args), tmp_dir=tmp_path / "second") == expected


def test_corrupt_dense_blob_is_repaired_and_deduplicated(tmp_path):
    base = np.arange(40000, dtype=np.float64).reshape(200, 200)
    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        store.save("cluster-e", "identity", base)
        store.save("preparation", "identity", (base, base))
        blobs = list((tmp_path / "state").glob("*.npy"))
        assert len(blobs) == 1
        blobs[0].write_bytes(b"truncated")
        assert store.load("preparation", "identity") is None
        store.save("cluster-e", "identity", base)
        restored = store.load("cluster-e", "identity")
        assert restored.tobytes() == base.tobytes()
        assert store.load("preparation", "identity")[0].tobytes() == base.tobytes()


def test_serializer_preserves_aliases_types_bytes_and_factories(tmp_path):
    base = np.arange(40000, dtype=">f8").reshape(200, 200)
    base[0, 0] = -0.
    dataset = DataSet(names=["A", "B"], data={"view": base[::-1]}, similarity=base,
                      cluster_similarity=base, cluster_names=None, weights=defaultdict(float, A=1.))
    dataset.cluster_names = dataset.names
    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        store.save("preparation", "test", ((dataset, {("A", 2): np.int32(4)}), state._rng()))
        (restored, mapping), rng = store.load("preparation", "test")
        assert restored.names is restored.cluster_names
        assert restored.similarity is restored.cluster_similarity
        assert np.shares_memory(restored.data["view"], restored.similarity)
        assert restored.similarity.dtype.str == base.dtype.str
        assert restored.similarity.tobytes() == base.tobytes()
        restored.data["view"][0, 0] = 99
        assert restored.similarity[-1, 0] == 99
        assert dataset.similarity[-1, 0] != 99
        assert restored.weights["missing"] == 0.
        assert type(mapping[("A", 2)]) is np.int32
        assert len(list((tmp_path / "state").glob("*.npy"))) == 1
        assert rng[0] == random.getstate()
        assert store.load("preparation", "test")[0][0].similarity[-1, 0] != 99


def test_fresh_process_restores_before_reader_overflow_or_clustering(tmp_path):
    script = tmp_path / "restart.py"
    script.write_text('''
import copy,pickle,sys
from pathlib import Path
from tests.test_preparation_checkpoints import fixture
from datasail.sail import datasail
import datasail.routine as routine
root=Path(sys.argv[1]);phase=sys.argv[2]
if phase=='interrupted':
    from datasail.solver import id_1d
    from datasail.solver.utils import solve
    def interrupted(*args,**kwargs):
        assert solve(*args,**kwargs) is not None
        raise KeyboardInterrupt('after native solver work')
    id_1d.solve=interrupted
else:
    def forbidden(*a,**kw):raise AssertionError('preparation stage reran')
    routine.read_data=forbidden
    routine.check_dataset=forbidden
    routine.cluster=forbidden
try:
    result=datasail(**copy.deepcopy(fixture()),tmp_dir=root/phase,checkpoint_dir=root/'checkpoint')
    (root/'result.pkl').write_bytes(pickle.dumps(result))
except KeyboardInterrupt:
    assert phase=='interrupted'
''')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for phase in ["interrupted", "resumed"]:
        subprocess.run([sys.executable, str(script), str(tmp_path), phase], env=env, check=True, capture_output=True, text=True)
    expected = datasail(**copy.deepcopy(fixture()), tmp_dir=tmp_path / "disabled")
    assert pickle.loads((tmp_path / "result.pkl").read_bytes()) == expected


@pytest.mark.parametrize("mode", ["assign", "break"])
def test_overflow_state_and_models_are_identical(tmp_path, monkeypatch, models, mode):
    from tests.test_c2_interaction_dispatch import public_fixture
    kwargs, _, _ = public_fixture()
    matrix = np.eye(20)
    matrix[:14, :14] = 1.
    kwargs.update(e_sim=(list(kwargs["e_data"]), matrix), overflow=mode, e_clusters=7,
                  techniques=["C2"], runs=2, epsilon=1., delta=1., max_sec=3)
    captured = []
    original = routine.run_solver

    def observe(**arguments):
        captured.append(state._signature((arguments["e_dataset"], arguments["f_dataset"],
                                           arguments["split_ratios"], arguments["split_names"])))
        return original(**arguments)

    monkeypatch.setattr(routine, "run_solver", observe)
    def execute(scratch, checkpoint=None):
        try:
            return datasail(**copy.deepcopy(kwargs), tmp_dir=scratch, checkpoint_dir=checkpoint)
        except KeyError as error:
            # Existing assign-overflow reconstruction can omit preassignments.
            # Preserve and compare this failure, never repair it in resume code.
            assert mode == "assign" and str(error) == "'L0'"
            return ("existing reconstruction KeyError", str(error))

    expected = execute(tmp_path / "disabled")
    expected_models = copy.deepcopy(models)
    models.clear()
    fresh = execute(tmp_path / "fresh", tmp_path / "checkpoint")
    assert models == expected_models
    models.clear()
    monkeypatch.setattr(routine, "check_dataset", lambda *a, **kw: pytest.fail("overflow ran twice"))
    resumed = execute(tmp_path / "resumed", tmp_path / "checkpoint")
    assert models == expected_models
    assert fresh == resumed == expected
    assert len(set(captured)) == 1


@pytest.mark.parametrize("change", ["linkage", "weights", "classes", "splits", "techniques", "inputs", "clusters"])
def test_changed_settings_reject_only_affected_stages(tmp_path, monkeypatch, change):
    from datasail.cluster import clustering
    original = clustering.run_ecfp
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(clustering, "run_ecfp", count)
    kwargs = fixture()
    datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "fresh", checkpoint_dir=tmp_path / "checkpoint")
    if change == "linkage":
        kwargs["linkage"] = "complete"
    elif change == "weights":
        kwargs["e_weights"] = dict.fromkeys(kwargs["e_data"], 2.)
    elif change == "classes":
        kwargs["e_strat"] = {n: {int(i % 3)} for i, n in enumerate(kwargs["e_data"])}
    elif change == "splits":
        kwargs["splits"] = [2, 1]
    elif change == "techniques":
        kwargs["techniques"] = list(reversed(kwargs["techniques"]))
    elif change == "inputs":
        kwargs["e_data"]["L0"] = "CCCO"
    else:
        kwargs["e_clusters"] = 3
    result = datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "modified", checkpoint_dir=tmp_path / "checkpoint")
    baseline = datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "baseline")
    assert result == baseline
    assert len(calls) == (3 if change == "inputs" else 2)  # baseline always computes once


def test_metadata_tampering_unsupported_state_and_scratch_paths(tmp_path):
    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        store.save("preparation", "identity", {"number": 3.})
        path = tmp_path / "state/preparation.json"
        record = state._read_json(path)
        record["values"]["nodes"][0][2][0][1] = 42
        state._publish_json(path, record)
        assert store.load("preparation", "identity") is None
        store.save("cluster-e", "identity", DataSet(data={"bad": object()}))
        assert not (tmp_path / "state/cluster-e.json").exists()
        store.save("cluster-e", "identity", DataSet(data={"bad": tmp_path / "scratch/deleted.pdb"}))
        assert not (tmp_path / "state/cluster-e.json").exists()


def test_foldseek_dense_file_is_linked_without_duplicate_storage(tmp_path):
    from datasail.cluster import _foldseek_checkpoint as foldseek
    native = tmp_path / "f"
    native.mkdir()
    base = np.arange(40000, dtype=np.float64).reshape(200, 200)
    # The native Foldseek builder returns an owning square matrix.
    base = base.copy()
    with foldseek._locked(native, {"test": "owned native checkpoint"}):
        foldseek._save_matrix(base, native, {}, [str(i) for i in range(200)])
    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        store.save("preparation", "identity", base)
        blob, = (tmp_path / "state").glob("*.npy")
        assert blob.stat().st_ino == (native / "similarity.npy").stat().st_ino
        restored = store.load("preparation", "identity")
        restored[0, 0] = -5
        assert np.load(native / "similarity.npy")[0, 0] == 0


@pytest.mark.parametrize("layout", ["fortran", "readonly", "independent", "empty"])
def test_array_layout_and_independent_equal_buffers(tmp_path, layout):
    base = np.arange(40000, dtype=np.float64).reshape(200, 200)
    if layout == "fortran":
        base = np.asfortranarray(base)
    if layout == "readonly":
        base.flags.writeable = False
    if layout == "empty":
        base = np.empty((0, 3))
    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        store.save("preparation", "identity", [base, base.copy()])
        a, b = store.load("preparation", "identity")
        assert a.dtype == base.dtype and a.shape == base.shape
        assert a.tobytes() == base.tobytes()
        assert a.strides == base.strides
        assert a.flags.writeable == base.flags.writeable
        assert not np.shares_memory(a, b)


def test_state_entry_concurrency_uses_existing_lock(tmp_path):
    from datasail.cluster._foldseek_checkpoint import _locked
    with state._using(tmp_path / "state", tmp_path / "scratch"):
        with pytest.raises(RuntimeError, match="in use"):
            with _locked(tmp_path / "state", {"kind": "preparation-state", "schema": 1}, allowed=state._allowed):
                pytest.fail("concurrent writer")


def test_ligand_failure_reuses_already_clustered_other_axis(tmp_path, monkeypatch):
    from datasail.cluster import clustering
    arguments = fixture()
    kwargs = {("f_" + k[2:] if k.startswith("e_") else "e_" + k[2:] if k.startswith("f_") else k): v
              for k, v in arguments.items()}
    kwargs.update(techniques=["C2"], runs=1, inter=[(f, e) for e, f in arguments["inter"]],
                  inter_strat={(f, e): v for (e, f), v in arguments["inter_strat"].items()})
    kwargs["checkpoint_dir"] = tmp_path / "checkpoint"
    calls = []
    ligand_calls = []
    cluster, ligand = clustering._cluster, clustering.run_ecfp

    def observed(ds, **args):
        calls.append(ds.type)
        return cluster(ds, **args)

    def interrupted(*args, **kw):
        ligand_calls.append(1)
        if len(ligand_calls) == 1:
            raise RuntimeError("ligand interrupted")
        return ligand(*args, **kw)

    monkeypatch.setattr(clustering, "_cluster", observed)
    monkeypatch.setattr(clustering, "run_ecfp", interrupted)
    with pytest.raises(RuntimeError, match="ligand interrupted"):
        datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "first")
    datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "restart")
    assert calls == ["O", "M", "M"]
    assert len(ligand_calls) == 2


def test_changed_file_contents_invalidate_preparation(tmp_path, monkeypatch):
    from datasail.cluster import clustering
    kwargs = fixture()
    source = tmp_path / "molecules.csv"
    source.write_text("name,smiles\n" + "".join(f"{k},{v}\n" for k, v in kwargs["e_data"].items()))
    kwargs["e_data"] = source
    counts = []
    compute = clustering.run_ecfp

    def observed(*args, **kw):
        counts.append(1)
        return compute(*args, **kw)

    monkeypatch.setattr(clustering, "run_ecfp", observed)
    datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "first", checkpoint_dir=tmp_path / "checkpoint")
    source.write_text(source.read_text().replace("L0,CC\n", "L0,CCCO\n"))
    actual = datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "second", checkpoint_dir=tmp_path / "checkpoint")
    expected = datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "disabled")
    assert actual == expected
    assert len(counts) == 3



def test_legacy_cache_contents_are_part_of_preparation_identity(tmp_path):
    cache = tmp_path / "legacy"
    cache.mkdir()
    kwargs = dict(fixture(), cache=True, cache_dir=cache)
    initial = state._request_identity(kwargs)
    (cache / "entry.pkl").write_bytes(b"changed cached scientific input")
    assert state._request_identity(kwargs) != initial
    kwargs["cache"] = False
    initial = state._request_identity(kwargs)
    (cache / "entry.pkl").write_bytes(b"irrelevant while legacy cache disabled")
    assert state._request_identity(kwargs) == initial



def test_custom_array_semantics_are_not_silently_coerced(tmp_path):
    class CustomArray(np.ndarray):
        pass

    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        store.save("preparation", "custom", np.arange(8).view(CustomArray))
        assert not (tmp_path / "state/preparation.json").exists()
        values = np.arange(8, dtype=np.dtype("f8", metadata={"custom": "semantics"}))
        store.save("preparation", "custom", values)
        assert not (tmp_path / "state/preparation.json").exists()



@pytest.mark.parametrize("dtype", [np.longlong, np.ulonglong, np.int64, np.float64, ">i8"])
def test_numpy_scalar_alias_types_and_array_dtypes_are_preserved(tmp_path, dtype):
    array = np.arange(12, dtype=dtype).reshape(3, 4)
    scalar = array[0, 0]
    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        store.save("preparation", "dtype", (scalar, array))
        restored_scalar, restored_array = store.load("preparation", "dtype")
        assert type(restored_scalar) is type(scalar)
        assert restored_array.dtype.type is array.dtype.type
        assert restored_array.dtype.str == array.dtype.str
        assert restored_array.tobytes() == array.tobytes()



@pytest.mark.parametrize("layout", ["C", "F", "one-dimensional", "empty"])
def test_streamed_array_verification_does_not_map_dense_storage(tmp_path, monkeypatch, layout):
    array = np.arange(40000, dtype=np.float64).reshape(200, 200)
    if layout == "F":
        array = np.asfortranarray(array)
    elif layout == "one-dimensional":
        array = array.ravel()
    elif layout == "empty":
        array = np.empty((0, 3))
    path = tmp_path / "test.npy"
    np.save(path, array)
    assert state._file_array_key(path) == state._array_key(array)
    with state._using(tmp_path / "state", tmp_path / "scratch") as store:
        record = store.array(array)
        monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("integrity check mapped dense storage"))
        assert store.array(array) == record



@pytest.mark.parametrize("filename", ["protein.PDB", "protein_without_extension"])
def test_foldseek_string_paths_hash_contents_independently_of_format_guess(tmp_path, filename):
    from datasail.reader.read_proteins import read_protein_data
    path = tmp_path / filename
    path.write_text("first structure\n")
    sources = {filename: str(path)}
    kwargs = dict(techniques=["C1f"], f_type="P", f_data=sources, f_sim="foldseek")
    dataset = read_protein_data(sources, sim="foldseek")
    request = state._request_identity(kwargs)
    axis = state._dataset_identity(dataset)
    path.write_text("other structure\n")
    assert state._request_identity(kwargs) != request
    assert state._dataset_identity(dataset) != axis



def test_reader_rng_consumption_cannot_be_rewound_by_cached_axes(tmp_path, monkeypatch):
    class AtSolver(Exception):
        pass

    observed = []

    def solver(**kwargs):
        observed.append(state._signature(state._rng()))
        raise AtSolver()

    monkeypatch.setattr(routine, "run_solver", solver)
    kwargs = fixture()
    kwargs["techniques"] = ["I1e", "C1f", "C2"]

    def weights(draws):
        def produce():
            np.random.random(draws)
            return dict.fromkeys(kwargs["e_data"], 1.)
        return produce

    for index, draws, checkpoint in [(0, 1, True), (1, 2, True), (2, 2, False)]:
        arguments = copy.deepcopy(kwargs)
        arguments["e_weights"] = weights(draws)
        with pytest.raises(AtSolver):
            datasail(**arguments, tmp_dir=tmp_path / str(index),
                     checkpoint_dir=tmp_path / "checkpoint" if checkpoint else None)
    assert observed[0] != observed[1]
    assert observed[1] == observed[2]

    folder = tmp_path / "checkpoint/state"
    record = state._read_json(folder / "cluster-e.json")
    _, _, coverage = state._unpack(record["values"], folder)
    assert "duplicate" in coverage["input_ids"]
    assert coverage["excluded_ids"] == ["invalid"]


def test_checkpoint_dir_public_api_preserves_positional_order(tmp_path, monkeypatch):
    import inspect
    parameters = inspect.signature(datasail).parameters
    assert list(parameters)[-3:] == ["threads", "checkpoint_dir", "foldseek_sort_memory"]
    assert "foldseek_checkpoint_dir" not in parameters
    kwargs = fixture()
    kwargs.update(techniques=["C2"], runs=1)
    expected = datasail(**copy.deepcopy(kwargs), tmp_dir=tmp_path / "disabled-positional")
    kwargs.update(tmp_dir=tmp_path / "fresh-positional", checkpoint_dir=tmp_path / "checkpoint")
    positional = [copy.deepcopy(kwargs.get(name, parameter.default)) for name, parameter in parameters.items()]
    assert datasail(*positional) == expected
    monkeypatch.setattr(routine, "_prepare", lambda *args: pytest.fail("preparation reran"))
    kwargs["tmp_dir"] = tmp_path / "resumed-positional"
    assert datasail(**kwargs) == expected


def test_state_entry_recovers_interrupted_ownership_write(tmp_path):
    directory = tmp_path / "state"
    directory.mkdir()
    (directory / ".lock").touch()
    (directory / "owner.json.part").write_text('{"format":')
    inode = (directory / ".lock").stat().st_ino
    with state._using(directory, tmp_path / "scratch") as store:
        store.save("preparation", "identity", {"complete": True})
    with state._using(directory, tmp_path / "different-scratch") as store:
        assert store.load("preparation", "identity") == {"complete": True}
    assert (directory / ".lock").stat().st_ino == inode
