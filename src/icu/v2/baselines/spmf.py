"""Faithful SPMF-backed sequential pattern and rule miners (v2 baselines).

The v1 baselines (``src/icu/comparators/run_prefixspan.py`` and ``run_rulegrowth.py``)
were capped approximations: contiguous 2-3-grams or adjacent 1->1 / 2->1 rules over the
first 40 events, with each pattern emitted once per threshold. v2 instead calls the
reference Java implementations in the official SPMF jar (P. Fournier-Viger et al.,
GPL-3.0) through its command line::

    java -jar spmf.jar run <Algorithm> <input> <output> <params...>

Pinned jar: SPMF 2.64b, ``spmf2.64.jar``, the versioned archive linked from the SPMF
download page (see ``tools/spmf/README.md`` for the URL and sha256). The newest release
(2.67a) is compiled for Java 25, while this machine runs Java 21. Positional parameters
differ between versions: the online documentation describes 2.67, where PrefixSpan and
CM-SPADE gained a "min pattern length" parameter. The lists below are the ones compiled
into 2.64b (``algorithmmanager/descriptions/DescriptionAlgo*.class``) and were checked by
running the jar on the documentation examples (``tests/v2/test_spmf.py``):

* ``PrefixSpan``: Minsup, [Max pattern length, Show sequence ids?]
* ``CM-SPADE``: Minsup, [Show sequence ids?]. This version has no length parameter, so
  :func:`mine_cmspade` drops longer patterns after mining. The result is the same
  pattern set, but the runtime is that of unconstrained CM-SPADE.
* ``RuleGrowth`` and ``ERMiner``: Minsup, Minconf, [Max antecedent size,
  Max consequent size]

The ``mine_*`` functions refuse other jar versions, because the same arguments would
change meaning. PrefixSpan and CM-SPADE are both complete miners of the frequent
sequential patterns, so they return the same set; CM-SPADE matters only as a runtime
comparator. TRuleGrowth is not wrapped: its window constraint cannot be expressed by the
shared program kinds in ``types.py``.

Behavior of SPMF 2.64b checked against brute-force enumeration
--------------------------------------------------------------
Checked on the documentation examples, on the unit-test databases, and in an opt-in
randomized sweep (``tests/v2/test_spmf.py``, set ``SPMF_SWEEP=1``; 150 random databases
during development):

* ``PrefixSpan``: exact in every case.
* ``CM-SPADE``: exact after canonicalization. It occasionally emits a pattern a second
  time with an unsorted itemset and a smaller support. Itemsets become frozensets here,
  and the larger support is kept, which matched brute force in every case.
* ``RuleGrowth``: INCOMPLETE. In ``AlgoRULEGROWTH.expandLeft``, the pruning step for a
  candidate item that can no longer become frequent jumps to the next itemset instead
  of the next item (bytecode: the branch to offset 261 increments the itemset index).
  Later items of the same itemset are then not counted, so rules whose antecedent has
  two or more items can be missed or have their support under-counted. 22 of 100
  random databases with a minimum count >= 2 were affected. Minimal case, minimum
  count 2: ``<(3 5)(2)>, <(4)>, <(4 5)(3)(2)>`` holds ``{3, 5} ==> {2}`` in two
  sequences, and RuleGrowth does not report it. ``expandRight`` is not affected.
* ``ERMiner`` (same rule definition and output, Fournier-Viger et al. 2014): exact for
  minimum counts >= 2 (100/100). At a minimum count of 1 it missed rules in 1 of 50
  databases, so :func:`mine_erminer` warns in that case.

Use :func:`mine_erminer` for the complete RuleGrowth rule set, or report the RuleGrowth
caveat if the cited implementation is used.

Protocol (v2)
-------------
1. Eventize every episode with ``events.eventize_episode``, using bins fitted on TRAIN.
2. Mine on the POSITIVE TRAIN sequences only (as the v1 baselines did), e.g.
   ``mine_prefixspan([seqs[e] for e in train_ids if labels[e] == 1], minsup=0.2)``.
3. Rank the mined programs with ``patterns.rank_by_contrast`` on TRAIN positives vs
   TRAIN controls, the same score for every symbolic miner.
4. Report minsup, the max pattern length, and the max antecedent/consequent sizes in
   the paper. Nothing else is capped: no event truncation, no top-k during mining, and
   each pattern is emitted once (``dedup_programs`` is applied).

Support is relative to all input sequences, including sequences without any token.
SPMF's file format cannot represent those, so they are not written. The minimum count
``ceil(minsup * n)`` is computed here and handed to SPMF as
``(count - 0.5) / n_written``, so it holds exactly whether an algorithm rounds with
``ceil`` or compares doubles. Low minsup values or long patterns can make the output
grow combinatorially; ``timeout_s`` bounds every call.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import warnings
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src.icu.v2.patterns import dedup_programs
from src.icu.v2.types import EventSequence, ProgramRecord, make_symbolic_program

SPMF_VERSION = "2.64b"
SPMF_JAR_URL = "https://www.philippe-fournier-viger.com/spmf/spmf2.64.jar"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPMF_JAR = REPO_ROOT / "tools" / "spmf" / "spmf.jar"
DEFAULT_TMP_ROOT = REPO_ROOT / "artifacts" / "v2" / "tmp_spmf"
DEFAULT_JVM_ARGS: tuple[str, ...] = ("-Xmx2g",)
SPMF_FAILURE_MARKER = "An error while trying to run the algorithm"
_USER_AGENT = "Mozilla/5.0 (compatible; fmg-bridge-v2-baselines)"
_SHA256_LINE = re.compile(r"^- sha256: `?([0-9a-f]{64})`?\s*$", re.MULTILINE)
_JAR_VERSION = re.compile(rb"SPMF version (\d+(?:\.\d+)+[a-z]?)")
_RULE_LINE = re.compile(
    r"^(?P<ante>.+?)\s*==>\s*(?P<cons>.+?)\s*#SUP:\s*(?P<sup>\d+)\s*#CONF:\s*(?P<conf>\S+)"
)


# ---------------------------------------------------------------------------- jar


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _recorded_sha256(readme: Path) -> str | None:
    if not readme.is_file():
        return None
    match = _SHA256_LINE.search(readme.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def spmf_jar_version(jar: str | Path = DEFAULT_SPMF_JAR) -> str | None:
    """Version compiled into the jar (``ca/pfv/spmf/gui/Main.class``), e.g. ``"2.64b"``."""
    path = Path(jar)
    stat = path.stat()
    return _jar_version_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=8)
def _jar_version_cached(path: str, mtime_ns: int, size: int) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            data = archive.read("ca/pfv/spmf/gui/Main.class")
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
    match = _JAR_VERSION.search(data)
    return match.group(1).decode("ascii") if match else None


def _write_readme(
    readme: Path,
    *,
    url: str,
    version: str,
    sha256: str,
    size: int,
    last_modified: str,
    downloaded_utc: str | None = None,
) -> None:
    downloaded = downloaded_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    readme.write_text(
        f"""# SPMF jar for the v2 baselines

`spmf.jar` is SPMF {version}, the official release jar of the open-source pattern mining
library by Philippe Fournier-Viger et al. (https://www.philippe-fournier-viger.com/spmf/,
GPL-3.0). It was downloaded from the versioned archive URL below by
`src/icu/v2/baselines/spmf.py::ensure_spmf_jar`, which pins it by sha256: if the jar is
missing, `ensure_spmf_jar` downloads it again and rejects a file with a different hash.

- version: {version} (the `SPMF version` string compiled into `ca/pfv/spmf/gui/Main.class`)
- url: {url}
- sha256: `{sha256}`
- size_bytes: {size}
- last_modified: {last_modified or "unknown"}
- downloaded_utc: {downloaded}

## Why 2.64b and not the newest release

The newest release at download time, 2.67a (2026-09-30, served at
https://www.philippe-fournier-viger.com/spmf/spmf.jar), is compiled for Java 25 (class
file version 69), and this machine runs Java 21 (class file version 65). 2.64b is
compiled for Java 21 and is the only versioned archive linked from the download page,
which also makes it reproducible. The online documentation describes 2.67, where
PrefixSpan and CM-SPADE gained a "min pattern length" parameter. The parameter lists
below are the ones compiled into 2.64b
(`ca/pfv/spmf/algorithmmanager/descriptions/DescriptionAlgo*.class`), and `spmf.py`
refuses other jar versions because positional parameters would change meaning.

## How the v2 code calls it

`java -Djava.awt.headless=true -Xmx2g -jar spmf.jar run <Algorithm> <input> <output> <params...>`

| Algorithm | Positional parameters in 2.64b (optional ones in brackets) | Documentation |
|---|---|---|
| `PrefixSpan` | Minsup (%), [Max pattern length, Show sequence ids?] | https://www.philippe-fournier-viger.com/spmf/PrefixSpan.php |
| `CM-SPADE` | Minsup (%), [Show sequence ids?] | https://www.philippe-fournier-viger.com/spmf/CM-SPADE.php |
| `RuleGrowth` | Minsup (%), Minconf (%), [Max antecedent size, Max consequent size] | https://www.philippe-fournier-viger.com/spmf/RuleGrowth.php |
| `ERMiner` | Minsup (%), Minconf (%), [Max antecedent size, Max consequent size] | https://www.philippe-fournier-viger.com/spmf/ERMiner.php |

- minsup and minconf are fractions in [0, 1] (`0.5` is the same as `50%`). Pattern
  lengths and rule sizes count items.
- CM-SPADE has no length parameter in 2.64b, so `mine_cmspade` drops longer patterns
  after mining. The pattern set is the same as PrefixSpan with that limit.
- Input: positive integer items, sorted ascending within an itemset; `-1` ends an itemset
  and `-2` ends a sequence. Output lines look like `1 2 -1 3 -1 #SUP: 4` (patterns) and
  `1,2 ==> 3 #SUP: 3 #CONF: 0.75` (rules). Version 2.64b writes no `@FILETYPE` header,
  and the parsers skip one if present.
- SPMF exits with code 0 when an algorithm fails. It prints "An error while trying to
  run the algorithm" and writes no output file, and `run_spmf` treats either sign as a
  failure.

## Known behavior of 2.64b (checked against brute-force enumeration)

- PrefixSpan: exact. CM-SPADE: exact after merging occasional duplicate patterns that it
  emits with an unsorted itemset and a smaller support (the larger support is kept).
- RuleGrowth is incomplete. The pruning step in `AlgoRULEGROWTH.expandLeft` jumps to the
  next itemset instead of the next item, so rules with two or more antecedent items can
  be missed or under-counted. 22 of 100 random databases with a minimum count >= 2
  were affected. Minimal case, minimum count 2: `<(3 5)(2)>, <(4)>, <(4 5)(3)(2)>`,
  where `{{3, 5}} ==> {{2}}` (support 2) is missed.
- ERMiner (same rule definition) is exact for minimum counts >= 2; at a minimum count
  of 1 it missed rules in 1 of 50 databases. `mine_erminer` returns the complete
  RuleGrowth rule set and is recommended for the rule baseline.
- `tests/v2/test_spmf.py` checks the documentation examples (53 patterns at minsup 50%;
  9 rules at minsup 75% / minconf 50%), brute-force agreement, and the RuleGrowth
  minimal case; `SPMF_SWEEP=1` enables the randomized sweep.
""",
        encoding="utf-8",
    )


def ensure_spmf_jar(
    dest: str | Path = DEFAULT_SPMF_JAR,
    *,
    url: str = SPMF_JAR_URL,
    expected_sha256: str | None = None,
    timeout_s: float = 300.0,
) -> Path:
    """Return the SPMF jar, downloading the pinned official release once if absent.

    The sha256 recorded in ``README.md`` next to ``dest`` (or ``expected_sha256``) pins
    the jar: an existing or newly downloaded file with a different hash raises. After a
    download, ``README.md`` records the version, URL, sha256, size, and Last-Modified.
    """
    dest = Path(dest)
    readme = dest.parent / "README.md"
    pinned = expected_sha256 or _recorded_sha256(readme)
    if dest.is_file():
        if pinned and _sha256(dest) != pinned:
            raise RuntimeError(f"{dest} does not match the pinned sha256 {pinned}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout_s) as response, open(partial, "wb") as handle:
            last_modified = response.headers.get("Last-Modified", "")
            shutil.copyfileobj(response, handle)
        digest = _sha256(partial)
        if pinned and digest != pinned:
            raise RuntimeError(f"downloaded jar has sha256 {digest}, expected {pinned}")
        if not zipfile.is_zipfile(partial):
            raise RuntimeError(f"{url} did not return a jar file")
        os.replace(partial, dest)
    finally:
        if partial.exists():
            partial.unlink()
    _write_readme(
        readme,
        url=url,
        version=spmf_jar_version(dest) or "unknown",
        sha256=digest,
        size=dest.stat().st_size,
        last_modified=last_modified,
    )
    return dest


# ------------------------------------------------------------------- file formats


def _vocabulary(sequences: Sequence[EventSequence]) -> list[str]:
    return sorted({token for seq in sequences for itemset in seq for token in itemset})


def write_spmf_sequence_db(
    sequences: Sequence[EventSequence],
    path: str | Path,
    token_to_int: dict[str, int] | None = None,
) -> dict[str, int]:
    """Write sequences in SPMF's sequence-database format; return token -> item id.

    Items are positive ints (1..V in sorted token order unless ``token_to_int`` is
    given), sorted ascending within an itemset; ``-1`` ends an itemset and ``-2`` a
    sequence. Empty itemsets are skipped, and so are sequences without any token
    (the format cannot represent them; the ``mine_*`` functions still count them).
    """
    if token_to_int is None:
        token_to_int = {token: index + 1 for index, token in enumerate(_vocabulary(sequences))}
    else:
        token_to_int = dict(token_to_int)
        ids = list(token_to_int.values())
        if any(not isinstance(i, int) or i < 1 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("token_to_int must map tokens to distinct positive ints")
        unknown = set(_vocabulary(sequences)) - set(token_to_int)
        if unknown:
            raise KeyError(f"tokens without an item id: {sorted(unknown)[:5]}")
    lines = []
    for seq in sequences:
        parts: list[str] = []
        for itemset in seq:
            if itemset:
                parts.extend(str(item) for item in sorted(token_to_int[token] for token in itemset))
                parts.append("-1")
        if parts:
            lines.append(" ".join(parts + ["-2"]))
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="ascii")
    return token_to_int


def _data_lines(path: str | Path) -> Iterator[str]:
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("@"):
            yield line


def parse_sequential_patterns(path: str | Path) -> list[tuple[tuple[tuple[int, ...], ...], int]]:
    """Parse SPMF pattern output (``1 2 -1 3 -1 #SUP: 4 [#SID: ...]``) into
    ``(itemsets, support_count)`` pairs; metadata lines starting with ``@`` are skipped."""
    patterns = []
    for line in _data_lines(path):
        body, marker, rest = line.partition("#SUP:")
        if not marker:
            raise ValueError(f"unrecognized SPMF pattern line: {line!r}")
        support = int(rest.split()[0])
        itemsets: list[tuple[int, ...]] = []
        current: list[int] = []
        for token in body.split():
            if token == "-1":
                itemsets.append(tuple(current))
                current = []
            elif token != "-2":
                current.append(int(token))
        if current:
            itemsets.append(tuple(current))
        patterns.append((tuple(itemsets), support))
    return patterns


def _items(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.split(r"[,\s]+", text.strip()) if item)


def parse_sequential_rules(path: str | Path) -> list[tuple[tuple[int, ...], tuple[int, ...], int, float]]:
    """Parse SPMF rule output (``1,2 ==> 3 #SUP: 3 #CONF: 0.75``) into
    ``(antecedent, consequent, support_count, confidence)`` tuples."""
    rules = []
    for line in _data_lines(path):
        match = _RULE_LINE.match(line)
        if not match:
            raise ValueError(f"unrecognized SPMF rule line: {line!r}")
        rules.append((_items(match["ante"]), _items(match["cons"]), int(match["sup"]), float(match["conf"])))
    return rules


# ------------------------------------------------------------------------ running


def run_spmf(
    algorithm: str,
    input_path: str | Path,
    output_path: str | Path,
    args: Sequence[str],
    timeout_s: float = 600,
    *,
    jar: str | Path | None = None,
    java: str = "java",
    jvm_args: Sequence[str] = DEFAULT_JVM_ARGS,
) -> str:
    """Run ``java -jar spmf.jar run <algorithm> <in> <out> <args...>`` and return stdout.

    The JVM runs headless (SPMF can never open its GUI) with the output directory as its
    working directory. SPMF exits with code 0 on algorithm errors, so a non-zero exit
    code, SPMF's error message, or a missing output file all raise ``RuntimeError``;
    ``subprocess.TimeoutExpired`` is raised after ``timeout_s`` seconds.
    """
    jar_path = Path(jar if jar is not None else DEFAULT_SPMF_JAR).resolve()
    if not jar_path.is_file():
        raise FileNotFoundError(f"SPMF jar not found at {jar_path}; call ensure_spmf_jar() first")
    java_exe = shutil.which(java)
    if java_exe is None:
        raise FileNotFoundError(f"java executable {java!r} not found on PATH")
    # Absolute paths: the JVM runs in the output directory, not in the caller's cwd.
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        output_path.unlink()
    command = [
        java_exe,
        "-Djava.awt.headless=true",
        *jvm_args,
        "-jar",
        str(jar_path),
        "run",
        algorithm,
        str(input_path),
        str(output_path),
        *(str(arg) for arg in args),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(output_path.parent),
    )
    log = completed.stdout + completed.stderr
    if completed.returncode != 0 or SPMF_FAILURE_MARKER in log or not output_path.is_file():
        raise RuntimeError(
            f"SPMF {algorithm} failed (exit code {completed.returncode}).\n"
            f"stdout: {completed.stdout[-2000:]}\nstderr: {completed.stderr[-2000:]}"
        )
    return completed.stdout


@contextmanager
def _workdir(prefix: str, tmp_root: str | Path | None) -> Iterator[Path]:
    root = Path(tmp_root if tmp_root is not None else DEFAULT_TMP_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def min_support_count(minsup: float, n_sequences: int) -> int:
    """Absolute minimum support ``max(1, ceil(minsup * n))`` with float-noise tolerance."""
    if not 0.0 < float(minsup) <= 1.0:
        raise ValueError("minsup must be in (0, 1]")
    return max(1, math.ceil(float(minsup) * int(n_sequences) - 1e-9))


def _format_double(value: float) -> str:
    return np.format_float_positional(float(value), unique=True, trim="-")


def _check_length(name: str, value: int | None) -> None:
    if value is not None and int(value) < 1:
        raise ValueError(f"{name} must be >= 1 or None")


def _check_jar(jar: str | Path | None) -> None:
    path = Path(jar) if jar is not None else DEFAULT_SPMF_JAR
    if not path.is_file():
        raise FileNotFoundError(f"SPMF jar not found at {path}; call ensure_spmf_jar() first")
    version = spmf_jar_version(path)
    if version != SPMF_VERSION:
        raise RuntimeError(
            f"{path} is SPMF {version}; the parameter layouts in spmf.py were verified for "
            f"SPMF {SPMF_VERSION} only"
        )


def _canonical(itemsets: Sequence[frozenset[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(itemset)) for itemset in itemsets)


def _mine_sequential(
    algorithm: str,
    method: str,
    sequences: Sequence[EventSequence],
    minsup: float,
    max_pattern_length: int | None,
    jar: str | Path | None,
    timeout_s: float,
    tmp_root: str | Path | None,
) -> list[ProgramRecord]:
    sequences = list(sequences)
    _check_length("max_pattern_length", max_pattern_length)
    min_count = min_support_count(minsup, len(sequences))
    n_written = sum(1 for seq in sequences if any(seq))
    if min_count > n_written:
        return []
    _check_jar(jar)
    with _workdir(algorithm, tmp_root) as workdir:
        db_path, out_path = workdir / "input.txt", workdir / "output.txt"
        mapping = write_spmf_sequence_db(sequences, db_path)
        args = [_format_double((min_count - 0.5) / n_written)]
        if max_pattern_length is not None and algorithm == "PrefixSpan":
            args.append(str(int(max_pattern_length)))
        run_spmf(algorithm, db_path, out_path, args, timeout_s, jar=jar)
        parsed = parse_sequential_patterns(out_path)
    token_of = {item: token for token, item in mapping.items()}
    supports: dict[tuple[frozenset[str], ...], int] = {}
    for itemsets, support in parsed:
        n_items = sum(len(itemset) for itemset in itemsets)
        if support < min_count or (max_pattern_length is not None and n_items > max_pattern_length):
            continue
        elements = tuple(frozenset(token_of[item] for item in itemset) for itemset in itemsets)
        supports[elements] = max(support, supports.get(elements, 0))  # CM-SPADE duplicates
    rows = sorted(supports.items(), key=lambda row: (-row[1], _canonical(row[0])))
    programs = [
        make_symbolic_program(
            f"{method}_{index:06d}",
            method,
            "sequence",
            elements,
            score=support / len(sequences),
            params={
                "algorithm": algorithm,
                "spmf_version": SPMF_VERSION,
                "support_count": support,
                "support": support / len(sequences),
                "n_sequences": len(sequences),
                "minsup": float(minsup),
                "min_support_count": min_count,
                "max_pattern_length": max_pattern_length,
            },
        )
        for index, (elements, support) in enumerate(rows)
    ]
    return dedup_programs(programs)


def mine_prefixspan(
    sequences: Sequence[EventSequence],
    minsup: float,
    max_pattern_length: int | None = None,
    *,
    jar: str | Path | None = None,
    timeout_s: float = 600.0,
    tmp_root: str | Path | None = None,
) -> list[ProgramRecord]:
    """All frequent sequential patterns via SPMF ``PrefixSpan`` (kind ``"sequence"``).

    ``minsup`` is a fraction of ``len(sequences)``; ``max_pattern_length`` counts items
    and is passed to SPMF. ``score`` and ``params["support"]`` hold the relative
    support; ids follow (support desc, canonical key).
    """
    return _mine_sequential("PrefixSpan", "prefixspan", sequences, minsup, max_pattern_length, jar, timeout_s, tmp_root)


def mine_cmspade(
    sequences: Sequence[EventSequence],
    minsup: float,
    max_pattern_length: int | None = None,
    *,
    jar: str | Path | None = None,
    timeout_s: float = 600.0,
    tmp_root: str | Path | None = None,
) -> list[ProgramRecord]:
    """All frequent sequential patterns via SPMF ``CM-SPADE`` (same contract as PrefixSpan).

    SPMF 2.64b's CM-SPADE has no length parameter, so ``max_pattern_length`` is applied
    after mining. The result is identical, but the run costs as much as an unconstrained
    one. Patterns that SPMF repeats with an unsorted itemset are merged, keeping the
    larger support (see the module docstring).
    """
    return _mine_sequential("CM-SPADE", "cmspade", sequences, minsup, max_pattern_length, jar, timeout_s, tmp_root)


def _mine_rules(
    algorithm: str,
    method: str,
    sequences: Sequence[EventSequence],
    minsup: float,
    minconf: float,
    max_antecedent: int | None,
    max_consequent: int | None,
    jar: str | Path | None,
    timeout_s: float,
    tmp_root: str | Path | None,
) -> list[ProgramRecord]:
    sequences = list(sequences)
    if not 0.0 <= float(minconf) <= 1.0:
        raise ValueError("minconf must be in [0, 1]")
    _check_length("max_antecedent", max_antecedent)
    _check_length("max_consequent", max_consequent)
    min_count = min_support_count(minsup, len(sequences))
    n_written = sum(1 for seq in sequences if any(seq))
    if min_count > n_written:
        return []
    _check_jar(jar)
    if algorithm == "ERMiner" and min_count == 1:
        warnings.warn(
            "SPMF 2.64b ERMiner can miss rules when the minimum support count is 1",
            UserWarning,
            stacklevel=3,
        )
    with _workdir(algorithm, tmp_root) as workdir:
        db_path, out_path = workdir / "input.txt", workdir / "output.txt"
        mapping = write_spmf_sequence_db(sequences, db_path)
        args = [_format_double((min_count - 0.5) / n_written), _format_double(minconf)]
        if max_antecedent is not None or max_consequent is not None:
            unbounded = str(len(mapping))  # no rule side can hold more items than exist
            args += [
                str(int(max_antecedent)) if max_antecedent is not None else unbounded,
                str(int(max_consequent)) if max_consequent is not None else unbounded,
            ]
        run_spmf(algorithm, db_path, out_path, args, timeout_s, jar=jar)
        parsed = parse_sequential_rules(out_path)
    token_of = {item: token for token, item in mapping.items()}
    found: dict[tuple[frozenset[str], frozenset[str]], tuple[int, float]] = {}
    for antecedent, consequent, support, confidence in parsed:
        if support < min_count:
            continue
        key = (frozenset(token_of[item] for item in antecedent), frozenset(token_of[item] for item in consequent))
        if key not in found or support > found[key][0]:
            found[key] = (support, confidence)
    rows = sorted(
        found.items(),
        key=lambda row: (-row[1][0], -row[1][1], tuple(sorted(row[0][0])), tuple(sorted(row[0][1]))),
    )
    programs = [
        make_symbolic_program(
            f"{method}_{index:06d}",
            method,
            "unordered_rule",
            (antecedent,),
            (consequent,),
            score=support / len(sequences),
            params={
                "algorithm": algorithm,
                "spmf_version": SPMF_VERSION,
                "support_count": support,
                "support": support / len(sequences),
                "confidence": confidence,
                "n_sequences": len(sequences),
                "minsup": float(minsup),
                "minconf": float(minconf),
                "min_support_count": min_count,
                "max_antecedent": max_antecedent,
                "max_consequent": max_consequent,
            },
        )
        for index, ((antecedent, consequent), (support, confidence)) in enumerate(rows)
    ]
    return dedup_programs(programs)


def mine_rulegrowth(
    sequences: Sequence[EventSequence],
    minsup: float,
    minconf: float,
    max_antecedent: int | None = None,
    max_consequent: int | None = None,
    *,
    jar: str | Path | None = None,
    timeout_s: float = 600.0,
    tmp_root: str | Path | None = None,
) -> list[ProgramRecord]:
    """Sequential rules X ==> Y via SPMF ``RuleGrowth`` (kind ``"unordered_rule"``).

    Support counts sequences where all items of X occur before all items of Y;
    confidence = support / #sequences containing all items of X. Each rule is stored as
    ``elements=(X,)`` and ``consequent=(Y,)``; ``score`` is the relative support and
    ``params`` hold the support count, support, and confidence.

    Caveat: SPMF 2.64b's RuleGrowth can miss rules whose antecedent has two or more
    items and can under-count their support (a defect in ``expandLeft``; see the module
    docstring). :func:`mine_erminer` returns the complete rule set for the same rule
    definition.
    """
    return _mine_rules(
        "RuleGrowth", "rulegrowth", sequences, minsup, minconf, max_antecedent, max_consequent,
        jar, timeout_s, tmp_root,
    )


def mine_erminer(
    sequences: Sequence[EventSequence],
    minsup: float,
    minconf: float,
    max_antecedent: int | None = None,
    max_consequent: int | None = None,
    *,
    jar: str | Path | None = None,
    timeout_s: float = 600.0,
    tmp_root: str | Path | None = None,
) -> list[ProgramRecord]:
    """The RuleGrowth rule set via SPMF ``ERMiner`` (Fournier-Viger et al., 2014).

    ERMiner mines exactly the rules that RuleGrowth defines, with the same parameters
    and output, so the contract is :func:`mine_rulegrowth`'s. It matched brute-force
    enumeration on every tested database with a minimum support count >= 2, and it warns
    when the count is 1.
    """
    return _mine_rules(
        "ERMiner", "erminer", sequences, minsup, minconf, max_antecedent, max_consequent,
        jar, timeout_s, tmp_root,
    )
