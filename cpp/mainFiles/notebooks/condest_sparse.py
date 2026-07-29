#!/usr/bin/env python3
"""
Memory-conscious 1-norm condition-number estimator for a large sparse matrix
stored as MATLAB-style triplets:

    row_index   column_index   value

Input may be a plain .dat/.txt file or a .zip archive containing one such file.

The estimate is the Python/SciPy analogue of MATLAB

    condest(A, t)

namely

    ||A||_1 * estimate(||A^{-1}||_1).

It is not guaranteed to be bit-for-bit identical to MATLAB, because SciPy and
MATLAB use different sparse factorizations/orderings and may make different
choices inside the norm estimator.

Memory strategy
---------------
* The text file is streamed in chunks; np.loadtxt/pandas are not used.
* Explicit zeros are dropped by default.
* The compressed sparse arrays are constructed directly on disk as .npy
  memory-mapped files. No full COO matrix is held in RAM.
* Only one sparse direct factorization is kept.
* `--backend pardiso` enables a parallel MKL PARDISO factorization when
  pypardiso is installed. `--backend scipy` uses SciPy/SuperLU, which is
  generally single-threaded.
* `--dry-run` scans the file and reports dimensions/storage without
  constructing or factorizing the matrix.

For a very large matrix, sparse LU fill-in can still require many times the
storage of the original sparse matrix. On Linux, run this program under a
systemd MemoryMax limit if you need a hard protection against an OOM freeze.


"""

from __future__ import annotations

# Thread variables should be set before NumPy/SciPy/MKL are imported.
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple


def _preparse_threads(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads", type=int, default=1)
    known, _ = parser.parse_known_args(argv)
    if known.threads < 1:
        raise SystemExit("--threads must be at least 1")
    return known.threads


_REQUESTED_THREADS = _preparse_threads(sys.argv[1:])
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = str(_REQUESTED_THREADS)

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, onenormest, splu


GIB = 1024**3
MIB = 1024**2


class MatrixFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputDescription:
    path: Path
    member: Optional[str]
    uncompressed_size: int
    fingerprint: dict


@dataclass
class ScanResult:
    n_rows: int
    n_cols: int
    n: int
    raw_entries: int
    kept_entries: int
    explicit_zeros: int
    major_counts: np.ndarray


class Progress:
    def __init__(self, label: str, total_bytes: int) -> None:
        self.label = label
        self.total_bytes = max(total_bytes, 1)
        self.started = time.monotonic()
        self.last_print = 0.0

    def update(self, consumed: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_print < 2.0:
            return
        self.last_print = now
        fraction = min(max(consumed / self.total_bytes, 0.0), 1.0)
        elapsed = max(now - self.started, 1e-9)
        rate = consumed / elapsed / MIB
        print(
            f"\r{self.label}: {100*fraction:6.2f}%  "
            f"{consumed/MIB:,.1f}/{self.total_bytes/MIB:,.1f} MiB  "
            f"{rate:,.1f} MiB/s",
            end="",
            flush=True,
        )
        if force:
            print()


def human_bytes(n: int | float) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:,.2f} {unit}"
        value /= 1024.0
    return f"{value:,.2f} TiB"


def linux_mem_available() -> Optional[int]:
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def describe_input(path: Path, requested_member: Optional[str]) -> InputDescription:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    stat = path.stat()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            files = [info for info in archive.infolist() if not info.is_dir()]
            if requested_member is not None:
                try:
                    info = archive.getinfo(requested_member)
                except KeyError as exc:
                    names = "\n".join(f"  {x.filename}" for x in files[:30])
                    raise MatrixFormatError(
                        f"{requested_member!r} is not in {path.name}.\n"
                        f"Archive members include:\n{names}"
                    ) from exc
            else:
                candidates = [
                    info
                    for info in files
                    if Path(info.filename).suffix.lower()
                    in {".dat", ".txt", ".coo", ".mtx"}
                ]
                if len(candidates) == 1:
                    info = candidates[0]
                elif len(files) == 1:
                    info = files[0]
                else:
                    names = "\n".join(f"  {x.filename}" for x in files[:30])
                    raise MatrixFormatError(
                        "The archive contains several files. Select one with "
                        f"--member NAME.\nArchive members include:\n{names}"
                    )
            fingerprint = {
                "kind": "zip",
                "archive_path": str(path),
                "archive_size": stat.st_size,
                "archive_mtime_ns": stat.st_mtime_ns,
                "member": info.filename,
                "member_size": info.file_size,
                "member_crc": info.CRC,
            }
            return InputDescription(
                path=path,
                member=info.filename,
                uncompressed_size=info.file_size,
                fingerprint=fingerprint,
            )

    fingerprint = {
        "kind": "plain",
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return InputDescription(
        path=path,
        member=None,
        uncompressed_size=stat.st_size,
        fingerprint=fingerprint,
    )


@contextlib.contextmanager
def open_input(desc: InputDescription) -> Iterator[BinaryIO]:
    if desc.member is None:
        with desc.path.open("rb") as stream:
            yield stream
    else:
        with zipfile.ZipFile(desc.path) as archive:
            with archive.open(desc.member, "r") as stream:
                yield stream


def _diagnose_bad_block(block: bytes, first_line_number: int) -> None:
    lines = block.splitlines()
    for offset, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 3:
            shown = line[:240].decode("ascii", errors="replace")
            raise MatrixFormatError(
                f"Malformed triplet at input line "
                f"{first_line_number + offset}: expected 3 columns, "
                f"found {len(fields)}.\n{shown}"
            )
        try:
            float(fields[0])
            float(fields[1])
            float(fields[2])
        except ValueError as exc:
            shown = line[:240].decode("ascii", errors="replace")
            raise MatrixFormatError(
                f"Non-numeric triplet at input line "
                f"{first_line_number + offset}:\n{shown}"
            ) from exc
    raise MatrixFormatError(
        f"Could not parse a block beginning near line {first_line_number}."
    )


def iter_triplet_chunks(
    stream: BinaryIO,
    chunk_bytes: int,
    total_bytes: int,
    label: str,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Yield arrays with shape (m, 3), parsed from complete lines.

    The second returned value is the number of uncompressed bytes consumed.
    """
    carry = b""
    line_number = 1
    consumed = 0
    progress = Progress(label, total_bytes)

    while True:
        raw = stream.read(chunk_bytes)
        if not raw:
            break
        consumed += len(raw)
        block = carry + raw
        last_newline = block.rfind(b"\n")
        if last_newline < 0:
            carry = block
            if len(carry) > 4 * chunk_bytes:
                raise MatrixFormatError(
                    "No newline found in a very large block; the input does "
                    "not appear to be a line-oriented triplet file."
                )
            progress.update(consumed)
            continue

        complete = block[: last_newline + 1]
        carry = block[last_newline + 1 :]
        number_of_lines = complete.count(b"\n")

        values = np.fromstring(complete, dtype=np.float64, sep=" ")
        expected = 3 * number_of_lines
        if values.size != expected:
            _diagnose_bad_block(complete, line_number)
        triplets = values.reshape(number_of_lines, 3)
        yield triplets, consumed

        line_number += number_of_lines
        progress.update(consumed)

    if carry.strip():
        values = np.fromstring(carry, dtype=np.float64, sep=" ")
        if values.size != 3:
            _diagnose_bad_block(carry, line_number)
        yield values.reshape(1, 3), consumed

    progress.update(total_bytes, force=True)


def validate_and_extract(
    triplets: np.ndarray,
    first_entry_number: int,
    drop_zeros: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    index_values = triplets[:, :2]
    values = triplets[:, 2]

    if not np.isfinite(index_values).all():
        raise MatrixFormatError(
            f"Non-finite row/column index near entry {first_entry_number}."
        )
    if not np.isfinite(values).all():
        raise MatrixFormatError(
            f"Non-finite matrix value near entry {first_entry_number}."
        )

    indices64 = index_values.astype(np.int64)
    if not np.array_equal(index_values, indices64.astype(np.float64)):
        raise MatrixFormatError(
            f"A non-integer row/column index occurs near entry "
            f"{first_entry_number}."
        )
    if indices64.min(initial=1) < 1:
        raise MatrixFormatError(
            f"Matrix indices must be 1-based positive integers; bad index "
            f"near entry {first_entry_number}."
        )

    rows = indices64[:, 0] - 1
    cols = indices64[:, 1] - 1

    if drop_zeros:
        keep = values != 0.0
        dropped = int(values.size - np.count_nonzero(keep))
        return rows[keep], cols[keep], values[keep], dropped
    return rows, cols, values, 0


def grow_counts(counts: np.ndarray, required: int) -> np.ndarray:
    if required <= counts.size:
        return counts
    new_size = max(required, max(1024, int(math.ceil(counts.size * 1.5))))
    new_counts = np.zeros(new_size, dtype=np.int64)
    new_counts[: counts.size] = counts
    return new_counts


def scan_input(
    desc: InputDescription,
    storage_format: str,
    chunk_bytes: int,
    drop_zeros: bool,
    forced_size: Optional[int],
) -> ScanResult:
    counts = np.zeros(0, dtype=np.int64)
    max_row = -1
    max_col = -1
    raw_entries = 0
    kept_entries = 0
    explicit_zeros = 0

    with open_input(desc) as stream:
        for triplets, _ in iter_triplet_chunks(
            stream,
            chunk_bytes,
            desc.uncompressed_size,
            "Pass 1/2: scanning",
        ):
            rows, cols, values, dropped = validate_and_extract(
                triplets,
                raw_entries + 1,
                drop_zeros,
            )
            raw_entries += triplets.shape[0]
            kept_entries += values.size
            explicit_zeros += dropped

            # Preserve the represented dimension even if the largest index
            # occurs only in an explicit-zero triplet that is dropped.
            max_row = max(max_row, int(triplets[:, 0].max()) - 1)
            max_col = max(max_col, int(triplets[:, 1].max()) - 1)

            if rows.size == 0:
                continue

            major = cols if storage_format == "csc" else rows
            required = int(major.max()) + 1
            counts = grow_counts(counts, required)
            counts[:required] += np.bincount(major, minlength=required)

    n_rows = max_row + 1
    n_cols = max_col + 1
    if forced_size is not None:
        if forced_size < max(n_rows, n_cols):
            raise MatrixFormatError(
                f"--size={forced_size} is smaller than an observed index "
                f"({max(n_rows, n_cols)})."
            )
        n_rows = n_cols = forced_size

    n = max(n_rows, n_cols)
    if n_rows != n_cols:
        raise MatrixFormatError(
            f"The represented matrix is not square: maximum row index is "
            f"{n_rows}, maximum column index is {n_cols}. If trailing rows or "
            f"columns are all zero, pass the intended square dimension using "
            f"--size N."
        )
    counts = grow_counts(counts, n)[:n]

    return ScanResult(
        n_rows=n_rows,
        n_cols=n_cols,
        n=n,
        raw_entries=raw_entries,
        kept_entries=kept_entries,
        explicit_zeros=explicit_zeros,
        major_counts=counts,
    )


def cache_metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "metadata.json"


def expected_cache_metadata(
    desc: InputDescription,
    storage_format: str,
    drop_zeros: bool,
    forced_size: Optional[int],
) -> dict:
    return {
        "version": 2,
        "source": desc.fingerprint,
        "format": storage_format,
        "drop_explicit_zeros": drop_zeros,
        "forced_size": forced_size,
    }


def load_cache(
    cache_dir: Path,
    expected: dict,
) -> Optional[Tuple[sp.spmatrix, dict]]:
    metadata_file = cache_metadata_path(cache_dir)
    required = [
        metadata_file,
        cache_dir / "data.npy",
        cache_dir / "indices.npy",
        cache_dir / "indptr.npy",
    ]
    if not all(path.is_file() for path in required):
        return None

    try:
        metadata = json.loads(metadata_file.read_text())
    except Exception:
        return None
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None

    data = np.load(cache_dir / "data.npy", mmap_mode="r+")
    indices = np.load(cache_dir / "indices.npy", mmap_mode="r+")
    indptr = np.load(cache_dir / "indptr.npy", mmap_mode="r+")
    n = int(metadata["n"])

    if metadata["format"] == "csc":
        matrix = sp.csc_matrix((data, indices, indptr), shape=(n, n), copy=False)
    else:
        matrix = sp.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
    return matrix, metadata


def build_cache(
    desc: InputDescription,
    scan: ScanResult,
    storage_format: str,
    chunk_bytes: int,
    drop_zeros: bool,
    cache_dir: Path,
    base_metadata: dict,
) -> Tuple[sp.spmatrix, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = scan.n
    nnz = scan.kept_entries
    if max(n, nnz) > np.iinfo(np.int32).max:
        raise MemoryError(
            "This script currently requires both the matrix dimension and "
            "number of stored entries to fit in signed 32-bit sparse indices."
        )

    required_disk = nnz * (8 + 4) + (n + 1) * 4
    free_disk = shutil.disk_usage(cache_dir).free
    if free_disk < int(required_disk * 1.15):
        raise MemoryError(
            f"Insufficient free disk space in {cache_dir}. The cache needs "
            f"about {human_bytes(required_disk)}, while "
            f"{human_bytes(free_disk)} is free."
        )

    indptr64 = np.empty(n + 1, dtype=np.int64)
    indptr64[0] = 0
    np.cumsum(scan.major_counts, out=indptr64[1:])
    if int(indptr64[-1]) != nnz:
        raise RuntimeError("Internal entry-count mismatch.")

    # int32 is enough for this matrix and is required by PARDISO/SuperLU.
    indptr = np.lib.format.open_memmap(
        cache_dir / "indptr.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n + 1,),
    )
    indptr[:] = indptr64
    indptr.flush()

    indices = np.lib.format.open_memmap(
        cache_dir / "indices.npy",
        mode="w+",
        dtype=np.int32,
        shape=(nnz,),
    )
    data = np.lib.format.open_memmap(
        cache_dir / "data.npy",
        mode="w+",
        dtype=np.float64,
        shape=(nnz,),
    )

    next_position = indptr64[:-1].copy()
    raw_entries = 0

    with open_input(desc) as stream:
        for triplets, _ in iter_triplet_chunks(
            stream,
            chunk_bytes,
            desc.uncompressed_size,
            "Pass 2/2: building",
        ):
            rows, cols, values, _ = validate_and_extract(
                triplets,
                raw_entries + 1,
                drop_zeros,
            )
            raw_entries += triplets.shape[0]
            if values.size == 0:
                continue

            if storage_format == "csc":
                major = cols
                minor = rows
            else:
                major = rows
                minor = cols

            # Group by the compressed dimension. For CSR input that is already
            # row sorted, this avoids sorting altogether.
            if major.size > 1 and np.any(major[1:] < major[:-1]):
                order = np.argsort(major, kind="stable")
                major = major[order]
                minor = minor[order]
                values = values[order]

            changes = np.empty(major.size, dtype=bool)
            changes[0] = True
            changes[1:] = major[1:] != major[:-1]
            starts = np.flatnonzero(changes)
            unique_major = major[starts]
            counts = np.diff(np.append(starts, major.size))

            group_id = np.cumsum(changes, dtype=np.int64) - 1
            local_offset = (
                np.arange(major.size, dtype=np.int64) - starts[group_id]
            )
            destinations = next_position[major] + local_offset

            indices[destinations] = minor.astype(np.int32, copy=False)
            data[destinations] = values
            next_position[unique_major] += counts

    if not np.array_equal(next_position, indptr64[1:]):
        raise RuntimeError("Internal compressed-array fill mismatch.")

    indices.flush()
    data.flush()

    metadata = dict(base_metadata)
    metadata.update(
        {
            "n": n,
            "raw_entries": scan.raw_entries,
            "nnz": nnz,
            "explicit_zeros_dropped": scan.explicit_zeros,
            "scipy_version": scipy.__version__,
        }
    )
    cache_metadata_path(cache_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    if storage_format == "csc":
        matrix = sp.csc_matrix((data, indices, indptr), shape=(n, n), copy=False)
    else:
        matrix = sp.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
    return matrix, metadata


def canonicalize(matrix: sp.spmatrix) -> Tuple[sp.spmatrix, int]:
    before = matrix.nnz
    matrix.sort_indices()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    removed = before - matrix.nnz
    return matrix, removed


def csr_one_norm(matrix: sp.csr_matrix, work_mb: int) -> float:
    n = matrix.shape[1]
    sums = np.zeros(n, dtype=np.float64)
    max_entries = max(1, work_mb * MIB // 16)
    for start in range(0, matrix.nnz, max_entries):
        stop = min(matrix.nnz, start + max_entries)
        chunk_indices = matrix.indices[start:stop]
        chunk_values = np.abs(matrix.data[start:stop])
        sums += np.bincount(
            chunk_indices,
            weights=chunk_values,
            minlength=n,
        )
    return float(sums.max(initial=0.0))


def csc_one_norm(matrix: sp.csc_matrix, work_mb: int) -> float:
    n = matrix.shape[1]
    max_entries = max(1, work_mb * MIB // 8)
    best = 0.0
    j = 0

    while j < n:
        data_start = int(matrix.indptr[j])
        target = min(matrix.nnz, data_start + max_entries)
        k = int(np.searchsorted(matrix.indptr, target, side="right") - 1)
        if k <= j:
            k = j + 1
        k = min(k, n)
        data_stop = int(matrix.indptr[k])

        lengths = np.diff(matrix.indptr[j : k + 1])
        nonempty = np.flatnonzero(lengths)
        if nonempty.size:
            absolute_values = np.abs(matrix.data[data_start:data_stop])
            starts = matrix.indptr[j:k][nonempty] - data_start
            column_sums = np.add.reduceat(absolute_values, starts)
            best = max(best, float(column_sums.max(initial=0.0)))
        j = k

    return best


def exact_sparse_one_norm(matrix: sp.spmatrix, work_mb: int) -> float:
    if sp.isspmatrix_csc(matrix):
        return csc_one_norm(matrix, work_mb)
    if sp.isspmatrix_csr(matrix):
        return csr_one_norm(matrix, work_mb)
    raise TypeError(f"Unsupported sparse format: {matrix.format}")


def empty_row_column_counts(matrix: sp.spmatrix) -> Tuple[int, int]:
    n = matrix.shape[0]
    if sp.isspmatrix_csr(matrix):
        empty_rows = int(np.count_nonzero(np.diff(matrix.indptr) == 0))
        col_counts = np.bincount(matrix.indices, minlength=n)
        empty_cols = int(np.count_nonzero(col_counts == 0))
    else:
        empty_cols = int(np.count_nonzero(np.diff(matrix.indptr) == 0))
        row_counts = np.bincount(matrix.indices, minlength=n)
        empty_rows = int(np.count_nonzero(row_counts == 0))
    return empty_rows, empty_cols


def scipy_inverse_operator(
    matrix: sp.csc_matrix,
    ordering: str,
) -> Tuple[LinearOperator, object]:
    print(f"Factoring with SciPy SuperLU; ordering={ordering} ...", flush=True)
    started = time.monotonic()
    lu = splu(
        matrix,
        permc_spec=ordering,
        options={"Equil": True},
    )
    print(f"Sparse LU completed in {time.monotonic() - started:,.1f} s.")

    n = matrix.shape[0]

    def solve_normal(x: np.ndarray) -> np.ndarray:
        return lu.solve(np.asarray(x, dtype=np.float64))

    def solve_transpose(x: np.ndarray) -> np.ndarray:
        return lu.solve(np.asarray(x, dtype=np.float64), trans="T")

    operator = LinearOperator(
        shape=(n, n),
        dtype=np.float64,
        matvec=solve_normal,
        rmatvec=solve_transpose,
        matmat=solve_normal,
        rmatmat=solve_transpose,
    )
    return operator, lu


def pardiso_inverse_operator(
    matrix: sp.csr_matrix,
) -> Tuple[LinearOperator, object]:
    try:
        from pypardiso import PyPardisoSolver
    except ImportError as exc:
        raise RuntimeError(
            "The PARDISO backend was requested but pypardiso is not "
            "installed. Install it with:\n"
            "  python -m pip install pypardiso"
        ) from exc

    # size_limit_storage=0 makes pypardiso store a hash rather than another
    # full Python copy of this large sparse matrix.
    solver = PyPardisoSolver(size_limit_storage=0)
    print(
        f"Factoring with MKL PARDISO using {_REQUESTED_THREADS} thread(s) ...",
        flush=True,
    )
    started = time.monotonic()
    solver.factorize(matrix)
    print(f"Sparse factorization completed in {time.monotonic() - started:,.1f} s.")

    n = matrix.shape[0]

    def _solve(x: np.ndarray, transpose: bool) -> np.ndarray:
        rhs = np.asarray(x, dtype=np.float64)
        if rhs.ndim == 2 and not rhs.flags.f_contiguous:
            rhs = np.asfortranarray(rhs)

        # pypardiso's public solve() resets iparm(12), so use its already
        # factorized phase directly. iparm(12)=1 requests A^T x=b for a CSR
        # matrix; 0 requests A x=b.
        solver.set_phase(33)
        solver.set_iparm(12, 1 if transpose else 0)
        return solver._call_pardiso(matrix, rhs)  # noqa: SLF001

    def solve_normal(x: np.ndarray) -> np.ndarray:
        return _solve(x, transpose=False)

    def solve_transpose(x: np.ndarray) -> np.ndarray:
        return _solve(x, transpose=True)

    operator = LinearOperator(
        shape=(n, n),
        dtype=np.float64,
        matvec=solve_normal,
        rmatvec=solve_transpose,
        matmat=solve_normal,
        rmatmat=solve_transpose,
    )
    return operator, solver


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pypardiso  # noqa: F401

        return "pardiso"
    except ImportError:
        return "scipy"


def default_cache_dir(input_path: Path, storage_format: str) -> Path:
    stem = input_path.name
    if input_path.suffix:
        stem = input_path.name[: -len(input_path.suffix)]
    return input_path.parent / f"{stem}_{storage_format}_cache"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the 1-norm condition number of a large sparse triplet "
            "matrix without loading the text file into memory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help=".dat file or .zip archive")
    parser.add_argument(
        "--member",
        help="member name when the zip archive contains more than one file",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "scipy", "pardiso"),
        default="auto",
        help=(
            "sparse direct solver; PARDISO is parallel, while SciPy SuperLU "
            "is normally single-threaded"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=_REQUESTED_THREADS,
        help="thread count for MKL/OpenMP numerical libraries",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=2,
        help=(
            "number of estimator columns; MATLAB condest uses t=2 by default"
        ),
    )
    parser.add_argument(
        "--itmax",
        type=int,
        default=5,
        help="maximum iterations used by scipy.sparse.linalg.onenormest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="NumPy random seed for repeatability",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=8,
        help="uncompressed text chunk size",
    )
    parser.add_argument(
        "--norm-work-mb",
        type=int,
        default=64,
        help="temporary memory used while computing the exact matrix 1-norm",
    )
    parser.add_argument(
        "--ordering",
        choices=("COLAMD", "MMD_AT_PLUS_A", "MMD_ATA", "NATURAL"),
        default="COLAMD",
        help="SuperLU column ordering; ignored by PARDISO",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "directory for memory-mapped compressed sparse arrays; by "
            "default it is created next to the input"
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="discard any compatible existing sparse-array cache",
    )
    parser.add_argument(
        "--keep-explicit-zeros",
        action="store_true",
        help="retain triplets whose value is exactly zero",
    )
    parser.add_argument(
        "--size",
        type=int,
        help=(
            "force a square dimension when trailing all-zero rows/columns are "
            "not represented in the triplet file"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and report matrix/storage information, then stop",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue despite conservative RAM warnings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON file in which to save the numerical result",
    )
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.t < 1:
        parser.error("--t must be at least 1")
    if args.itmax < 2:
        parser.error("--itmax must be at least 2")
    if args.chunk_mb < 1:
        parser.error("--chunk-mb must be at least 1")
    if args.norm_work_mb < 1:
        parser.error("--norm-work-mb must be at least 1")
    if args.size is not None and args.size < 1:
        parser.error("--size must be positive")
    return args


def main() -> int:
    args = parse_arguments()
    np.random.seed(args.seed)

    desc = describe_input(args.input, args.member)
    backend = choose_backend(args.backend)
    storage_format = "csr" if backend == "pardiso" else "csc"
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else default_cache_dir(desc.path, storage_format)
    )

    print(f"Input:       {desc.path}")
    if desc.member is not None:
        print(f"Zip member:  {desc.member}")
    print(f"Backend:     {backend}")
    print(f"Threads:     {args.threads}")
    print(f"Cache:       {cache_dir}")
    print(f"Text size:   {human_bytes(desc.uncompressed_size)}")

    base_metadata = expected_cache_metadata(
        desc,
        storage_format,
        not args.keep_explicit_zeros,
        args.size,
    )

    cached = None
    if not args.rebuild_cache:
        cached = load_cache(cache_dir, base_metadata)

    scan: Optional[ScanResult] = None
    if cached is None or args.dry_run:
        scan = scan_input(
            desc=desc,
            storage_format=storage_format,
            chunk_bytes=args.chunk_mb * MIB,
            drop_zeros=not args.keep_explicit_zeros,
            forced_size=args.size,
        )
        sparse_bytes = (
            scan.kept_entries * (8 + 4) + (scan.n + 1) * 4
        )
        estimator_bytes = scan.n * args.t * 8 * 8
        print(f"Dimension:   {scan.n:,} x {scan.n:,}")
        print(f"Triplets:    {scan.raw_entries:,}")
        print(f"Kept nnz:    {scan.kept_entries:,}")
        print(f"Zero entries dropped: {scan.explicit_zeros:,}")
        print(f"CSC/CSR array storage: about {human_bytes(sparse_bytes)}")
        print(
            f"Approx. norm-estimator workspace: "
            f"{human_bytes(estimator_bytes)}"
        )

        available = linux_mem_available()
        if available is not None:
            print(f"Currently available RAM: {human_bytes(available)}")
            # This only protects the loading/estimator workspace. LU fill-in
            # is matrix dependent and cannot be bounded from the triplets.
            conservative_minimum = sparse_bytes + estimator_bytes + 2 * GIB
            if available < conservative_minimum and not args.force:
                raise MemoryError(
                    "Available RAM is already low for the matrix plus working "
                    "arrays. Close other applications or pass --force. This "
                    "check does not include unpredictable sparse-LU fill-in."
                )

        if args.dry_run:
            print(
                "\nDry run complete. No sparse matrix was built and no "
                "factorization was attempted."
            )
            return 0

    if cached is not None:
        matrix, metadata = cached
        print(
            f"Loaded memory-mapped {metadata['format'].upper()} cache: "
            f"{metadata['n']:,} x {metadata['n']:,}, "
            f"{metadata['nnz']:,} stored entries."
        )
    else:
        assert scan is not None
        if args.rebuild_cache and cache_dir.exists():
            shutil.rmtree(cache_dir)
        matrix, metadata = build_cache(
            desc=desc,
            scan=scan,
            storage_format=storage_format,
            chunk_bytes=args.chunk_mb * MIB,
            drop_zeros=not args.keep_explicit_zeros,
            cache_dir=cache_dir,
            base_metadata=base_metadata,
        )
        print(
            f"Built memory-mapped {storage_format.upper()} cache with "
            f"{matrix.nnz:,} entries."
        )

    matrix, removed = canonicalize(matrix)
    if removed:
        print(
            f"Canonicalization combined duplicates/removed cancellations: "
            f"{removed:,} entries removed."
        )

    empty_rows, empty_cols = empty_row_column_counts(matrix)
    if empty_rows or empty_cols:
        print(
            f"The matrix has {empty_rows:,} empty row(s) and "
            f"{empty_cols:,} empty column(s); it is singular."
        )
        estimate = float("inf")
        result = {
            "condition_estimate_1_norm": estimate,
            "matrix_one_norm": None,
            "inverse_one_norm_estimate": None,
            "n": matrix.shape[0],
            "nnz": matrix.nnz,
            "backend": backend,
            "threads": args.threads,
            "t": args.t,
            "itmax": args.itmax,
        }
        if args.output:
            args.output.write_text(json.dumps(result, indent=2))
        print("condest-like estimate: inf")
        return 0

    print("Computing exact sparse matrix 1-norm ...", flush=True)
    started = time.monotonic()
    matrix_norm_1 = exact_sparse_one_norm(matrix, args.norm_work_mb)
    print(
        f"||A||_1 = {matrix_norm_1:.16e} "
        f"({time.monotonic() - started:,.1f} s)"
    )

    try:
        if backend == "scipy":
            if not sp.isspmatrix_csc(matrix):
                matrix = matrix.tocsc()
            inverse_operator, factorization = scipy_inverse_operator(
                matrix,
                args.ordering,
            )
        else:
            if not sp.isspmatrix_csr(matrix):
                matrix = matrix.tocsr()
            inverse_operator, factorization = pardiso_inverse_operator(matrix)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "singular" in message or "exactly singular" in message:
            print(f"Factorization reports a singular matrix: {exc}")
            print("condest-like estimate: inf")
            return 0
        raise

    print(
        f"Estimating ||A^(-1)||_1 with t={args.t}, "
        f"itmax={args.itmax} ...",
        flush=True,
    )
    started = time.monotonic()
    inverse_norm_estimate = float(
        onenormest(
            inverse_operator,
            t=args.t,
            itmax=args.itmax,
        )
    )
    elapsed_estimator = time.monotonic() - started
    condition_estimate = matrix_norm_1 * inverse_norm_estimate

    print(f"estimated ||A^(-1)||_1 = {inverse_norm_estimate:.16e}")
    print(f"condest-like estimate    = {condition_estimate:.16e}")
    print(f"Norm estimation time     = {elapsed_estimator:,.1f} s")

    result = {
        "condition_estimate_1_norm": condition_estimate,
        "matrix_one_norm": matrix_norm_1,
        "inverse_one_norm_estimate": inverse_norm_estimate,
        "n": matrix.shape[0],
        "nnz": matrix.nnz,
        "backend": backend,
        "threads": args.threads,
        "t": args.t,
        "itmax": args.itmax,
        "seed": args.seed,
        "scipy_version": scipy.__version__,
        "input": str(desc.path),
        "zip_member": desc.member,
    }
    if args.output:
        args.output.expanduser().resolve().write_text(
            json.dumps(result, indent=2, sort_keys=True)
        )
        print(f"Saved result to {args.output}")

    # Keep references alive until after the estimate, then explicitly release
    # PARDISO's native memory where possible.
    if backend == "pardiso":
        try:
            factorization.free_memory(everything=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except (MemoryError, MatrixFormatError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
