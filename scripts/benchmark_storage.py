"""Benchmark candidate on-disk storage formats for self-interferometry shot data.

This script is an *evaluation* tool: it does NOT touch the production data
pipeline. It loads a representative subset of real acquisition data from the
current HDF5 store, writes that subset out in every candidate format whose
library is installed in the active environment, and measures four metrics per
format:

    * write time      -- wall-clock to serialize the subset to disk
    * on-disk size     -- total bytes of the produced file(s)
    * full-load time   -- wall-clock to read every channel fully into NumPy
    * row-read time     -- mean wall-clock to fetch ONE random shot (one row
                          across all channels), averaged over many random rows.
                          This mirrors the PyTorch DataLoader access pattern,
                          where ``__getitem__`` pulls a single shot.

Run with::

    pixi run -e dev python scripts/benchmark_storage.py

Missing libraries are skipped gracefully (detected via ``importlib``) and
reported at the end. Each format runs inside its own try/except so a single
failure cannot abort the whole run. All temporary outputs are written under a
scratch directory and removed afterwards.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import shutil
import signal
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Absolute path to a small real dataset (~157 MB). Subsetting keeps memory sane.
SOURCE_H5 = Path(
    '/Users/nolanpeard/Documents/Projects/self-interferometry/'
    'smi/analysis/data/circuit-noise-600.h5'
)

# Channels (datasets) present in every file; one row across all = one "shot".
CHANNELS = ('RP1_CH1', 'RP1_CH2', 'RP2_CH1', 'RP2_CH2')

# Representative subset: first N_SHOTS rows x len(CHANNELS) channels.
# Kept small so the whole benchmark runs in well under a minute; the metrics
# are relative and stable at this size.
N_SHOTS = 200

# Number of random single-row reads to average for the row-access metric.
N_RANDOM_READS = 50

# Hard per-candidate wall-clock cap. A candidate that exceeds this is recorded
# as 'timeout' and skipped so one pathological format cannot hang the run.
PER_CANDIDATE_TIMEOUT_S = 30

# Scratch directory for temporary outputs (never committed). Defaults to a
# gitignored directory beside the source data on the real disk; override with
# the SMI_BENCH_SCRATCH environment variable. Avoid small RAM-backed tmpfs
# locations (e.g. some system temp dirs) -- the format conversions are large.
SCRATCH_DIR = Path(
    os.environ.get('SMI_BENCH_SCRATCH', str(SOURCE_H5.parent / '.bench_scratch'))
)


class _CandidateTimeout(Exception):
    """Raised by the SIGALRM handler when a candidate exceeds its time cap."""


def _on_alarm(signum: int, frame: object) -> None:
    raise _CandidateTimeout


def have(module: str) -> bool:
    """Return True if ``module`` can be imported in this environment."""
    return importlib.util.find_spec(module) is not None


@dataclass
class Result:
    """Measured metrics for one storage format (or a skip/error record)."""

    name: str
    write_s: float = float('nan')
    size_mb: float = float('nan')
    full_load_s: float = float('nan')
    row_read_ms: float = float('nan')
    status: str = 'ok'
    notes: str = ''


def _dir_size_bytes(path: Path) -> int:
    """Total size in bytes of a file or all files under a directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def _timed(fn: Callable[[], object]) -> tuple[float, object]:
    """Run ``fn`` once and return (elapsed_seconds, return_value)."""
    gc.disable()
    try:
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
    finally:
        gc.enable()
    return elapsed, out


# ---------------------------------------------------------------------------
# Data loading (source of truth)
# ---------------------------------------------------------------------------


def load_subset() -> dict[str, np.ndarray]:
    """Load the first N_SHOTS rows of every channel from the source HDF5 file.

    Returns:
        Mapping channel-name -> (N_SHOTS, n_samples) float32 array.
    """
    import h5py

    data: dict[str, np.ndarray] = {}
    with h5py.File(SOURCE_H5, 'r') as f:
        for ch in CHANNELS:
            data[ch] = np.asarray(f[ch][:N_SHOTS], dtype=np.float32)
    return data


# ---------------------------------------------------------------------------
# Per-format benchmark functions
# ---------------------------------------------------------------------------
#
# Each function takes the in-memory ``data`` dict and an output path/dir,
# writes the data, then measures full-load and single-row-read timing.
# Write time and size are measured by the harness around the write call.


def bench_hdf5(
    data: dict[str, np.ndarray],
    out: Path,
    *,
    compression: str | int | None,
    compression_opts: object = None,
    chunks: object = None,
    label: str,
    plugin_filter: object = None,
) -> Result:
    """Benchmark an HDF5 variant (h5py).

    Official docs: https://docs.h5py.org/en/stable/high/dataset.html
    """
    import h5py

    n_shots = data[CHANNELS[0]].shape[0]
    res = Result(name=label)

    def write() -> None:
        with h5py.File(out, 'w') as f:
            for ch, arr in data.items():
                kwargs: dict[str, object] = {}
                if plugin_filter is not None:
                    # hdf5plugin filter object (e.g. Blosc) carries its own opts.
                    kwargs.update(plugin_filter)
                    if chunks is not None:
                        kwargs['chunks'] = chunks
                else:
                    if compression is not None:
                        kwargs['compression'] = compression
                    if compression_opts is not None:
                        kwargs['compression_opts'] = compression_opts
                    if chunks is not None:
                        kwargs['chunks'] = chunks
                f.create_dataset(ch, data=arr, **kwargs)

    res.write_s, _ = _timed(write)
    res.size_mb = _dir_size_bytes(out) / 1e6

    def full_load() -> None:
        with h5py.File(out, 'r') as f:
            for ch in CHANNELS:
                _ = f[ch][:]

    def row_read(i: int) -> None:
        with h5py.File(out, 'r') as f:
            for ch in CHANNELS:
                _ = f[ch][i]

    res.full_load_s = _measure_full_load(full_load)
    res.row_read_ms = _measure_row_reads(row_read, n_shots)
    return res


def bench_npy_memmap(data: dict[str, np.ndarray], out: Path) -> Result:
    """Benchmark a stack of per-channel .npy files read via mmap.

    One .npy per channel; reads use np.load(mmap_mode='r') so a single row is
    fetched without loading the whole array.
    Official docs: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
    """
    out.mkdir(parents=True, exist_ok=True)
    n_shots = data[CHANNELS[0]].shape[0]
    res = Result(name='npy (mmap, uncompressed)')

    def write() -> None:
        for ch, arr in data.items():
            np.save(out / f'{ch}.npy', arr)

    res.write_s, _ = _timed(write)
    res.size_mb = _dir_size_bytes(out) / 1e6

    def full_load() -> None:
        for ch in CHANNELS:
            _ = np.load(out / f'{ch}.npy')

    def row_read(i: int) -> None:
        for ch in CHANNELS:
            mm = np.load(out / f'{ch}.npy', mmap_mode='r')
            _ = np.asarray(mm[i])
            del mm

    res.full_load_s = _measure_full_load(full_load)
    res.row_read_ms = _measure_row_reads(row_read, n_shots)
    return res


def bench_npz(data: dict[str, np.ndarray], out: Path, *, compressed: bool) -> Result:
    """Benchmark a single .npz archive (compressed or not).

    npz has no row-level random access: each row read must open the archive and
    decompress/extract the full channel array. We measure that honestly.
    Official docs: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
    """
    n_shots = data[CHANNELS[0]].shape[0]
    label = 'npz (compressed)' if compressed else 'npz (uncompressed)'
    res = Result(name=label)
    saver = np.savez_compressed if compressed else np.savez

    def write() -> None:
        saver(out, **data)

    res.write_s, _ = _timed(write)
    # np.savez appends .npz if missing.
    npz_path = out if out.suffix == '.npz' else out.with_suffix('.npz')
    res.size_mb = _dir_size_bytes(npz_path) / 1e6

    def full_load() -> None:
        with np.load(npz_path) as z:
            for ch in CHANNELS:
                _ = z[ch]

    def row_read(i: int) -> None:
        with np.load(npz_path) as z:
            for ch in CHANNELS:
                _ = z[ch][i]

    res.full_load_s = _measure_full_load(full_load)
    res.row_read_ms = _measure_row_reads(row_read, n_shots)
    return res


def bench_zarr(
    data: dict[str, np.ndarray], out: Path, *, label: str, compressors: object
) -> Result:
    """Benchmark a Zarr v3 store, one array per channel, chunked per shot.

    Chunk shape (1, n_samples) => one chunk per shot, so a single-row read
    decompresses exactly one shot. Official docs:
    https://zarr.readthedocs.io/en/stable/user-guide/arrays/
    """
    import zarr

    n_shots, n_samples = data[CHANNELS[0]].shape
    res = Result(name=label)

    def write() -> None:
        store = zarr.open_group(str(out), mode='w')
        for ch, arr in data.items():
            z = store.create_array(
                name=ch,
                shape=arr.shape,
                chunks=(1, n_samples),
                dtype=arr.dtype,
                compressors=compressors,
            )
            z[:] = arr

    res.write_s, _ = _timed(write)
    res.size_mb = _dir_size_bytes(out) / 1e6

    def full_load() -> None:
        store = zarr.open_group(str(out), mode='r')
        for ch in CHANNELS:
            _ = store[ch][:]

    def row_read(i: int) -> None:
        store = zarr.open_group(str(out), mode='r')
        for ch in CHANNELS:
            _ = store[ch][i]

    res.full_load_s = _measure_full_load(full_load)
    res.row_read_ms = _measure_row_reads(row_read, n_shots)
    return res


def bench_parquet(
    data: dict[str, np.ndarray], out: Path, *, compression: str
) -> Result:
    """Benchmark Apache Parquet via pyarrow.

    Layout: one column per channel, each cell a fixed-length list of n_samples
    floats; one row == one shot. Official docs:
    https://arrow.apache.org/docs/python/parquet.html
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_shots, n_samples = data[CHANNELS[0]].shape
    res = Result(name=f'parquet ({compression}, pyarrow)')

    # Build a table where each channel is a list<float32> column.
    def to_list_array(arr: np.ndarray) -> pa.Array:
        flat = pa.array(arr.reshape(-1), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, n_samples)

    table = pa.table({ch: to_list_array(arr) for ch, arr in data.items()})

    def write() -> None:
        pq.write_table(table, out, compression=compression)

    res.write_s, _ = _timed(write)
    res.size_mb = _dir_size_bytes(out) / 1e6

    def full_load() -> None:
        t = pq.read_table(out)
        for ch in CHANNELS:
            _ = np.asarray(t[ch].combine_chunks().values).reshape(-1)

    # Random single-row access: read just one row group / slice.
    pf = pq.ParquetFile(out)

    def row_read(i: int) -> None:
        # Parquet has no native row index; reading a single logical row still
        # requires touching (and decoding) its whole row group.
        t = pf.read_row_group(0, columns=list(CHANNELS))
        for ch in CHANNELS:
            _ = t[ch][i]

    res.full_load_s = _measure_full_load(full_load)
    res.row_read_ms = _measure_row_reads(row_read, n_shots)
    return res


def bench_feather(
    data: dict[str, np.ndarray], out: Path, *, compression: str
) -> Result:
    """Benchmark Arrow IPC / Feather V2 via pyarrow.

    Official docs: https://arrow.apache.org/docs/python/feather.html
    """
    import pyarrow as pa
    from pyarrow import feather

    n_shots, n_samples = data[CHANNELS[0]].shape
    res = Result(name=f'feather ({compression}, arrow IPC)')

    def to_list_array(arr: np.ndarray) -> pa.Array:
        flat = pa.array(arr.reshape(-1), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, n_samples)

    table = pa.table({ch: to_list_array(arr) for ch, arr in data.items()})

    def write() -> None:
        feather.write_feather(table, out, compression=compression)

    res.write_s, _ = _timed(write)
    res.size_mb = _dir_size_bytes(out) / 1e6

    def full_load() -> None:
        t = feather.read_table(out)
        for ch in CHANNELS:
            _ = np.asarray(t[ch].combine_chunks().values).reshape(-1)

    # Memory-map + zero-copy slice for single-row reads.
    def row_read(i: int) -> None:
        with pa.memory_map(str(out), 'r') as source:
            t = feather.read_table(source)
            for ch in CHANNELS:
                _ = t[ch][i]

    res.full_load_s = _measure_full_load(full_load)
    res.row_read_ms = _measure_row_reads(row_read, n_shots)
    return res


def bench_sqlite(data: dict[str, np.ndarray], out: Path) -> Result:
    """Benchmark SQLite with one BLOB per (shot, channel) -- 64 KiB blobs.

    Each shot is 16384 float32 = 64 KiB, near SQLite's "store internally"
    sweet spot. Official docs: https://www.sqlite.org/intern-v-extern-blob.html
    """
    import sqlite3

    n_shots = data[CHANNELS[0]].shape[0]
    res = Result(name='sqlite (BLOB rows)')

    def write() -> None:
        con = sqlite3.connect(out)
        con.execute('PRAGMA page_size=16384')
        con.execute('PRAGMA journal_mode=OFF')
        con.execute(
            'CREATE TABLE shots (shot INTEGER, channel TEXT, data BLOB, '
            'PRIMARY KEY (shot, channel))'
        )
        rows = [
            (i, ch, data[ch][i].tobytes()) for i in range(n_shots) for ch in CHANNELS
        ]
        con.executemany('INSERT INTO shots VALUES (?, ?, ?)', rows)
        con.commit()
        con.close()

    res.write_s, _ = _timed(write)
    res.size_mb = _dir_size_bytes(out) / 1e6

    def full_load() -> None:
        con = sqlite3.connect(out)
        for ch in CHANNELS:
            cur = con.execute(
                'SELECT data FROM shots WHERE channel=? ORDER BY shot', (ch,)
            )
            for (blob,) in cur:
                _ = np.frombuffer(blob, dtype=np.float32)
        con.close()

    def row_read(i: int) -> None:
        con = sqlite3.connect(out)
        cur = con.execute('SELECT channel, data FROM shots WHERE shot=?', (i,))
        for _ch, blob in cur:
            _ = np.frombuffer(blob, dtype=np.float32)
        con.close()

    res.full_load_s = _measure_full_load(full_load)
    res.row_read_ms = _measure_row_reads(row_read, n_shots)
    return res


def _shot_arrow_table(data: dict[str, np.ndarray]) -> object:
    """Build a pyarrow table: a 'shot' index column + one FixedSizeList<f32>
    column per channel. Used for fast (zero-copy) DuckDB ingestion -- inserting
    16384-element python lists row by row is pathologically slow.
    """
    import pyarrow as pa

    n_shots, n_samples = data[CHANNELS[0]].shape

    def to_list_array(arr: np.ndarray) -> object:
        flat = pa.array(arr.reshape(-1), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, n_samples)

    columns = {'shot': pa.array(np.arange(n_shots), type=pa.int32())}
    columns.update({ch: to_list_array(arr) for ch, arr in data.items()})
    return pa.table(columns)


def bench_duckdb(
    data: dict[str, np.ndarray], out: Path, *, over_parquet: Path | None
) -> Result:
    """Benchmark DuckDB, either native .duckdb or querying a Parquet file.

    Official docs: https://duckdb.org/docs/current/data/parquet/overview.html
    Layout: one row per shot, a FLOAT[] list column per channel. Data is ingested
    from an in-memory Arrow table (DuckDB reads Arrow zero-copy).
    """
    import duckdb
    import pyarrow.parquet as pq

    n_shots = data[CHANNELS[0]].shape[0]
    mode = 'over parquet' if over_parquet is not None else 'native'
    res = Result(name=f'duckdb ({mode})')
    arrow_tbl = _shot_arrow_table(data)

    if over_parquet is not None:

        def write() -> None:
            pq.write_table(arrow_tbl, over_parquet, compression='zstd')

        res.write_s, _ = _timed(write)
        res.size_mb = _dir_size_bytes(over_parquet) / 1e6

        # Reuse a single connection across reads -- a fresh duckdb.connect() per
        # query pays full engine-startup overhead and dominates the measurement.
        read_con = duckdb.connect()

        def full_load() -> None:
            read_con.execute(f"SELECT * FROM read_parquet('{over_parquet}')").fetchall()

        def row_read(i: int) -> None:
            read_con.execute(
                f"SELECT * FROM read_parquet('{over_parquet}') WHERE shot=?", [i]
            ).fetchall()
    else:

        def write() -> None:
            con = duckdb.connect(str(out))
            con.register('arrow_tbl', arrow_tbl)
            con.execute('CREATE TABLE shots AS SELECT * FROM arrow_tbl')
            con.unregister('arrow_tbl')
            con.close()

        res.write_s, _ = _timed(write)
        res.size_mb = _dir_size_bytes(out) / 1e6

        read_con = duckdb.connect(str(out), read_only=True)

        def full_load() -> None:
            read_con.execute('SELECT * FROM shots').fetchall()

        def row_read(i: int) -> None:
            read_con.execute('SELECT * FROM shots WHERE shot=?', [i]).fetchall()

    try:
        res.full_load_s = _measure_full_load(full_load)
        res.row_read_ms = _measure_row_reads(row_read, n_shots)
    finally:
        read_con.close()
    return res


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _measure_full_load(full_load: Callable[[], None]) -> float:
    """Best-of-3 full-load time (takes the min to reduce noise)."""
    times = []
    for _ in range(3):
        t, _ = _timed(full_load)
        times.append(t)
    return min(times)


def _measure_row_reads(row_read: Callable[[int], None], n_shots: int) -> float:
    """Mean milliseconds per single random-row read over N_RANDOM_READS draws."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_shots, size=N_RANDOM_READS)
    # Warm up.
    row_read(int(idx[0]))
    t0 = time.perf_counter()
    for i in idx:
        row_read(int(i))
    return (time.perf_counter() - t0) / N_RANDOM_READS * 1e3


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A benchmark candidate: required modules + a runner closure."""

    name: str
    required: tuple[str, ...]
    run: Callable[[Path], Result]
    cleanup_paths: list[Path] = field(default_factory=list)


def build_candidates(data: dict[str, np.ndarray], scratch: Path) -> list[Candidate]:
    """Assemble the list of candidate formats to attempt."""
    n_samples = data[CHANNELS[0]].shape[1]
    candidates: list[Candidate] = []

    # --- HDF5 baseline and tuned variants -------------------------------
    p = scratch / 'baseline_gzip4_chunked.h5'
    candidates.append(
        Candidate(
            'hdf5 (gzip-4, chunked) [baseline]',
            ('h5py',),
            lambda out, p=p: bench_hdf5(
                data,
                p,
                compression='gzip',
                compression_opts=4,
                chunks=(1, n_samples),
                label='hdf5 (gzip-4, chunked) [baseline]',
            ),
            [p],
        )
    )

    p = scratch / 'hdf5_lzf.h5'
    candidates.append(
        Candidate(
            'hdf5 (lzf, chunked)',
            ('h5py',),
            lambda out, p=p: bench_hdf5(
                data,
                p,
                compression='lzf',
                chunks=(1, n_samples),
                label='hdf5 (lzf, chunked)',
            ),
            [p],
        )
    )

    p = scratch / 'hdf5_gzip9.h5'
    candidates.append(
        Candidate(
            'hdf5 (gzip-9, chunked)',
            ('h5py',),
            lambda out, p=p: bench_hdf5(
                data,
                p,
                compression='gzip',
                compression_opts=9,
                chunks=(1, n_samples),
                label='hdf5 (gzip-9, chunked)',
            ),
            [p],
        )
    )

    p = scratch / 'hdf5_uncompressed.h5'
    candidates.append(
        Candidate(
            'hdf5 (uncompressed, chunked)',
            ('h5py',),
            lambda out, p=p: bench_hdf5(
                data,
                p,
                compression=None,
                chunks=(1, n_samples),
                label='hdf5 (uncompressed, chunked)',
            ),
            [p],
        )
    )

    # hdf5plugin Blosc filter (zstd+shuffle) if available.
    def _hdf5_blosc(out: Path, p: Path) -> Result:
        import hdf5plugin

        flt = dict(
            hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        return bench_hdf5(
            data,
            p,
            compression=None,
            chunks=(1, n_samples),
            label='hdf5 (blosc-zstd+shuffle, chunked)',
            plugin_filter=flt,
        )

    p = scratch / 'hdf5_blosc.h5'
    candidates.append(
        Candidate(
            'hdf5 (blosc-zstd+shuffle, chunked)',
            ('h5py', 'hdf5plugin'),
            lambda out, p=p: _hdf5_blosc(out, p),
            [p],
        )
    )

    # --- NumPy ----------------------------------------------------------
    p = scratch / 'npy_store'
    candidates.append(
        Candidate(
            'npy (mmap, uncompressed)',
            ('numpy',),
            lambda out, p=p: bench_npy_memmap(data, p),
            [p],
        )
    )

    p = scratch / 'data_uncompressed'
    candidates.append(
        Candidate(
            'npz (uncompressed)',
            ('numpy',),
            lambda out, p=p: bench_npz(data, p, compressed=False),
            [p.with_suffix('.npz')],
        )
    )

    p = scratch / 'data_compressed'
    candidates.append(
        Candidate(
            'npz (compressed)',
            ('numpy',),
            lambda out, p=p: bench_npz(data, p, compressed=True),
            [p.with_suffix('.npz')],
        )
    )

    # --- Zarr -----------------------------------------------------------
    def _zarr_zstd(out: Path, p: Path) -> Result:
        from zarr.codecs import BloscCodec

        return bench_zarr(
            data,
            p,
            label='zarr (blosc-zstd, per-shot chunks)',
            compressors=BloscCodec(cname='zstd', clevel=5, shuffle='shuffle'),
        )

    p = scratch / 'zarr_zstd.zarr'
    candidates.append(
        Candidate(
            'zarr (blosc-zstd, per-shot chunks)',
            ('zarr',),
            lambda out, p=p: _zarr_zstd(out, p),
            [p],
        )
    )

    # --- Parquet --------------------------------------------------------
    for comp in ('zstd', 'snappy'):
        p = scratch / f'data_{comp}.parquet'
        candidates.append(
            Candidate(
                f'parquet ({comp}, pyarrow)',
                ('pyarrow',),
                lambda out, p=p, c=comp: bench_parquet(data, p, compression=c),
                [p],
            )
        )

    # --- Feather / Arrow IPC -------------------------------------------
    for comp in ('zstd', 'lz4', 'uncompressed'):
        p = scratch / f'data_{comp}.feather'
        candidates.append(
            Candidate(
                f'feather ({comp}, arrow IPC)',
                ('pyarrow',),
                lambda out, p=p, c=comp: bench_feather(data, p, compression=c),
                [p],
            )
        )

    # --- SQLite ---------------------------------------------------------
    p = scratch / 'data.sqlite'
    candidates.append(
        Candidate(
            'sqlite (BLOB rows)',
            ('sqlite3',),
            lambda out, p=p: bench_sqlite(data, p),
            [p],
        )
    )

    # --- DuckDB ---------------------------------------------------------
    p = scratch / 'data.duckdb'
    candidates.append(
        Candidate(
            'duckdb (native)',
            ('duckdb',),
            lambda out, p=p: bench_duckdb(data, p, over_parquet=None),
            [p],
        )
    )
    pq_path = scratch / 'duckdb_over.parquet'
    candidates.append(
        Candidate(
            'duckdb (over parquet)',
            ('duckdb',),
            lambda out, p=pq_path: bench_duckdb(data, p, over_parquet=p),
            [pq_path],
        )
    )

    return candidates


def main() -> None:
    """Run the benchmark and print a comparison table."""
    print('=' * 78)
    print('Storage-format benchmark for self-interferometry shot data')
    print('=' * 78)
    print(f'Source : {SOURCE_H5}')
    print(
        f'Subset : first {N_SHOTS} shots x {len(CHANNELS)} channels '
        f'({", ".join(CHANNELS)})'
    )
    print(f'Random-row reads averaged over {N_RANDOM_READS} draws')
    print()

    if not SOURCE_H5.exists():
        print(f'ERROR: source file not found: {SOURCE_H5}')
        print(
            'Real datasets are gitignored and absent from worktrees. Run from '
            'the main working tree.'
        )
        return

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading subset into memory...')
    data = load_subset()
    n_shots, n_samples = data[CHANNELS[0]].shape
    raw_mb = sum(a.nbytes for a in data.values()) / 1e6
    print(
        f'Loaded {n_shots} shots x {n_samples} samples x {len(CHANNELS)} '
        f'channels = {raw_mb:.1f} MB raw float32 in memory.'
    )
    print()

    signal.signal(signal.SIGALRM, _on_alarm)

    candidates = build_candidates(data, SCRATCH_DIR)
    results: list[Result] = []
    skipped: list[tuple[str, str]] = []

    for cand in candidates:
        missing = [m for m in cand.required if not have(m)]
        if missing:
            msg = f'missing {", ".join(missing)}'
            skipped.append((cand.name, msg))
            results.append(Result(name=cand.name, status='skipped', notes=msg))
            print(f'[skip] {cand.name}: {msg}')
            continue
        try:
            print(f'[run ] {cand.name} ...', flush=True)
            t0 = time.perf_counter()
            signal.alarm(PER_CANDIDATE_TIMEOUT_S)
            res = cand.run(SCRATCH_DIR)
            signal.alarm(0)
            results.append(res)
            print(f'[ ok ] {cand.name} ({time.perf_counter() - t0:.1f}s)', flush=True)
        except _CandidateTimeout:
            note = f'>{PER_CANDIDATE_TIMEOUT_S}s wall-clock cap'
            results.append(Result(name=cand.name, status='timeout', notes=note))
            print(f'[TIME] {cand.name}: {note}', flush=True)
        except Exception as exc:  # noqa: BLE001 -- isolate per-format failures
            results.append(Result(name=cand.name, status='error', notes=str(exc)))
            print(f'[FAIL] {cand.name}: {exc}', flush=True)
        finally:
            signal.alarm(0)
            for path in cand.cleanup_paths:
                _remove(path)

    _print_table(results, raw_mb)

    if skipped:
        print()
        print('Skipped (missing libraries):')
        for name, why in skipped:
            print(f'  - {name}: {why}')


def _remove(path: Path) -> None:
    """Delete a temp file or directory if present."""
    with suppress(FileNotFoundError):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _print_table(results: list[Result], raw_mb: float) -> None:
    """Print an aligned comparison table sorted by row-read time."""
    print()
    print('=' * 100)
    print(f'RESULTS (raw in-memory size: {raw_mb:.1f} MB)')
    print('=' * 100)
    header = (
        f'{"format":<38}{"write_s":>9}{"size_MB":>9}'
        f'{"compress":>9}{"load_s":>9}{"row_ms":>9}'
    )
    print(header)
    print('-' * len(header))

    ok = [r for r in results if r.status == 'ok']
    other = [r for r in results if r.status != 'ok']
    ok.sort(key=lambda r: r.row_read_ms if r.row_read_ms == r.row_read_ms else 1e9)

    for r in ok:
        ratio = (
            raw_mb / r.size_mb if r.size_mb and r.size_mb == r.size_mb else float('nan')
        )
        print(
            f'{r.name:<38}{r.write_s:>9.3f}{r.size_mb:>9.1f}'
            f'{ratio:>8.2f}x{r.full_load_s:>9.3f}{r.row_read_ms:>9.3f}'
        )

    for r in other:
        print(
            f'{r.name:<38}{"--":>9}{"--":>9}{"--":>9}{"--":>9}{"--":>9}'
            f'   [{r.status}: {r.notes}]'
        )

    print('-' * len(header))
    print(
        'write_s = serialize time | size_MB = on-disk | compress = raw/disk '
        'ratio | load_s = full read | row_ms = mean ms per random single-row read'
    )


if __name__ == '__main__':
    main()
