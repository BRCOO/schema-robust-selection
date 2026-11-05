from __future__ import annotations

import contextlib
import hashlib
import itertools
import os
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from src.icu.v2.baselines import spmf
from src.icu.v2.baselines.spmf import (
    DEFAULT_SPMF_JAR,
    DEFAULT_TMP_ROOT,
    SPMF_VERSION,
    ensure_spmf_jar,
    min_support_count,
    mine_cmspade,
    mine_erminer,
    mine_prefixspan,
    mine_rulegrowth,
    parse_sequential_patterns,
    parse_sequential_rules,
    run_spmf,
    spmf_jar_version,
    write_spmf_sequence_db,
)
from src.icu.v2.patterns import fires
from src.icu.v2.types import make_symbolic_program

HAVE_SPMF = shutil.which("java") is not None and DEFAULT_SPMF_JAR.is_file()
requires_spmf = pytest.mark.skipif(not HAVE_SPMF, reason="java or tools/spmf/spmf.jar unavailable")

# contextPrefixSpan.txt from the SPMF documentation (PrefixSpan / CM-SPADE / RuleGrowth pages).
DOC_DB_LINES = [
    "1 -1 1 2 3 -1 1 3 -1 4 -1 3 6 -1 -2",
    "1 4 -1 3 -1 2 3 -1 1 5 -1 -2",
    "5 6 -1 1 2 -1 4 6 -1 3 -1 2 -1 -2",
    "5 -1 7 -1 1 6 -1 3 -1 2 -1 3 -1 -2",
]
TOKENS = ["HR:q0", "HR:q1", "Lactate:q3", "Lactate:missing", "vasopressor_on:on"]


def doc_db() -> list[tuple[frozenset[str], ...]]:
    db = []
    for line in DOC_DB_LINES:
        parts = [part.split() for part in line.replace("-2", "").split("-1")]
        db.append(tuple(frozenset(f"i{item}" for item in part) for part in parts if part))
    return db


def random_db(seed: int, n: int = 8, alphabet: list[str] = TOKENS) -> list[tuple[frozenset[str], ...]]:
    rng = random.Random(seed)
    return [
        tuple(frozenset(rng.sample(alphabet, rng.randint(1, 3))) for _ in range(rng.randint(1, 5)))
        for _ in range(n)
    ]


def brute_force_patterns(db, min_count: int, max_items: int | None = None) -> dict:
    """Every sequential pattern (tuple of itemsets) contained in >= min_count sequences."""
    counts: Counter = Counter()
    for seq in db:
        found: set = set()

        def extend(start: int, prefix: tuple, n_items: int) -> None:
            for pos in range(start, len(seq)):
                items = sorted(seq[pos])
                for size in range(1, len(items) + 1):
                    if max_items is not None and n_items + size > max_items:
                        break
                    for combo in itertools.combinations(items, size):
                        pattern = prefix + (frozenset(combo),)
                        if pattern not in found:  # first discovery has the earliest end
                            found.add(pattern)
                            extend(pos + 1, pattern, n_items + size)

        extend(0, (), 0)
        counts.update(found)
    return {pattern: count for pattern, count in counts.items() if count >= min_count}


def brute_force_rules(db, min_count: int, minconf: float, max_ante=None, max_cons=None) -> dict:
    """Every rule X ==> Y (disjoint, non-empty) with RuleGrowth support and confidence."""
    alphabet = sorted({token for seq in db for itemset in seq for token in itemset})
    contents = [frozenset().union(*seq) for seq in db]

    def subsets(pool, limit):
        for size in range(1, (limit or len(pool)) + 1):
            for combo in itertools.combinations(pool, size):
                yield frozenset(combo)

    rules = {}
    for antecedent in subsets(alphabet, max_ante):
        antecedent_count = sum(1 for items in contents if antecedent <= items)
        if antecedent_count < min_count:
            continue
        rest = [token for token in alphabet if token not in antecedent]
        for consequent in subsets(rest, max_cons):
            program = make_symbolic_program("bf", "bf", "unordered_rule", (antecedent,), (consequent,))
            support = sum(fires(program, seq) for seq in db)
            if support >= min_count and support / antecedent_count >= minconf:
                rules[(antecedent, consequent)] = (support, support / antecedent_count)
    return rules


def pattern_supports(programs) -> dict:
    return {program.elements: program.params["support_count"] for program in programs}


@pytest.fixture
def scratch():
    DEFAULT_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="pytest_", dir=DEFAULT_TMP_ROOT))
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------------------ pure Python helpers


def test_write_db_format_and_mapping(scratch):
    seqs = [
        (frozenset({"b:q1", "a:q0"}), frozenset(), frozenset({"c:on"})),
        (),
        (frozenset(),),
        (frozenset({"a:q0"}),),
    ]
    path = scratch / "db.txt"
    mapping = write_spmf_sequence_db(seqs, path)
    assert mapping == {"a:q0": 1, "b:q1": 2, "c:on": 3}
    assert path.read_text(encoding="ascii").splitlines() == ["1 2 -1 3 -1 -2", "1 -1 -2"]

    write_spmf_sequence_db(seqs[:1], path, token_to_int={"a:q0": 9, "b:q1": 4, "c:on": 1})
    assert path.read_text(encoding="ascii").splitlines() == ["4 9 -1 1 -1 -2"]
    with pytest.raises(KeyError):
        write_spmf_sequence_db(seqs, path, token_to_int={"a:q0": 1})
    with pytest.raises(ValueError):
        write_spmf_sequence_db(seqs, path, token_to_int={"a:q0": 1, "b:q1": 1, "c:on": 2})
    with pytest.raises(ValueError):
        write_spmf_sequence_db(seqs, path, token_to_int={"a:q0": 0, "b:q1": 1, "c:on": 2})
    assert write_spmf_sequence_db([], path) == {} and path.read_text() == ""


def test_token_mapping_round_trip(scratch):
    db = random_db(5, n=20) + [(), (frozenset(),)]
    path = scratch / "db.txt"
    mapping = write_spmf_sequence_db(db, path)
    token_of = {item: token for token, item in mapping.items()}
    decoded = []
    for line in path.read_text(encoding="ascii").splitlines():
        assert line.endswith("-1 -2")
        itemsets = [part.split() for part in line[: -len("-2")].split("-1")]
        decoded.append(tuple(frozenset(token_of[int(item)] for item in part) for part in itemsets if part))
        for part in itemsets:
            assert [int(i) for i in part] == sorted(int(i) for i in part)
    assert decoded == [tuple(s for s in seq if s) for seq in db if any(seq)]
    assert sorted(mapping.values()) == list(range(1, len(mapping) + 1))


def test_parse_pattern_and_rule_output(scratch):
    patterns = scratch / "patterns.txt"
    patterns.write_text(
        '@FILETYPE="Frequent sequential patterns"\n'
        '@SOURCE="SPMF SOFTWARE https://philippe-fournier-viger.com/spmf/"\n'
        "2 3 -1 1 -1 #SUP: 2\n"
        "6 -1 2 -1 #SUP: 2 #SID: 0 2\n"
        "\n"
        "7 -1 #SUP: 1\n",
        encoding="utf-8",
    )
    assert parse_sequential_patterns(patterns) == [
        (((2, 3), (1,)), 2),
        (((6,), (2,)), 2),
        (((7,),), 1),
    ]
    rules = scratch / "rules.txt"
    rules.write_text(
        '@FILETYPE="Frequent sequential rules"\n'
        "1,2 ==> 3 #SUP: 3 #CONF: 1.0\n"
        "1 ==> 2,3 #SUP: 4 #CONF: 0.75\n",
        encoding="utf-8",
    )
    assert parse_sequential_rules(rules) == [((1, 2), (3,), 3, 1.0), ((1,), (2, 3), 4, 0.75)]
    (scratch / "bad.txt").write_text("1 -1 2 -1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_sequential_patterns(scratch / "bad.txt")
    with pytest.raises(ValueError):
        parse_sequential_rules(scratch / "bad.txt")
    (scratch / "empty.txt").write_text("", encoding="utf-8")
    assert parse_sequential_patterns(scratch / "empty.txt") == []


def test_min_support_count():
    assert min_support_count(0.3, 10) == 3  # 0.3 * 10 = 3.0000000000000004 in floating point
    assert min_support_count(0.5, 4) == 2
    assert min_support_count(0.5, 5) == 3
    assert min_support_count(0.01, 10) == 1
    assert min_support_count(1.0, 7) == 7
    assert min_support_count(0.25, 0) == 1
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            min_support_count(bad, 10)


def test_ensure_spmf_jar_respects_pin_without_downloading(scratch):
    payload = b"stand-in for a jar"
    jar = scratch / "spmf.jar"
    jar.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (scratch / "README.md").write_text(f"- sha256: `{digest}`\n", encoding="utf-8")
    assert ensure_spmf_jar(jar) == jar
    (scratch / "README.md").write_text(f"- sha256: `{'0' * 64}`\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ensure_spmf_jar(jar)
    assert ensure_spmf_jar(jar, expected_sha256=digest) == jar


def test_jar_version_detection_and_guard(scratch):
    fake = scratch / "fake.jar"
    with zipfile.ZipFile(fake, "w") as archive:
        archive.writestr("ca/pfv/spmf/gui/Main.class", b"\x01\x00-- SPMF version 9.99z --\x00")
    assert spmf_jar_version(fake) == "9.99z"
    (scratch / "plain.jar").write_bytes(b"not a zip")
    assert spmf_jar_version(scratch / "plain.jar") is None
    with pytest.raises(RuntimeError, match="verified for SPMF"):
        mine_prefixspan(doc_db(), 0.5, jar=fake)
    with pytest.raises(FileNotFoundError):
        mine_rulegrowth(doc_db(), 0.5, 0.5, jar=scratch / "missing.jar")


@pytest.mark.skipif(not DEFAULT_SPMF_JAR.is_file(), reason="tools/spmf/spmf.jar unavailable")
def test_pinned_jar_matches_readme():
    readme = DEFAULT_SPMF_JAR.parent / "README.md"
    assert ensure_spmf_jar() == DEFAULT_SPMF_JAR  # no download: hash checked against README
    assert spmf._recorded_sha256(readme) == spmf._sha256(DEFAULT_SPMF_JAR)
    assert spmf_jar_version(DEFAULT_SPMF_JAR) == SPMF_VERSION


# ------------------------------------------------------------------- SPMF runs


@requires_spmf
def test_documentation_example_sequential_patterns():
    db = doc_db()
    expected = brute_force_patterns(db, 2)
    assert len(expected) == 53  # "53 sequential patterns are found" (SPMF docs)
    for miner, method in ((mine_prefixspan, "prefixspan"), (mine_cmspade, "cmspade")):
        programs = miner(db, 0.5)
        assert pattern_supports(programs) == expected
        assert len(programs) == 53 and len({p.canonical_key() for p in programs}) == 53
        assert all(p.kind == "sequence" and p.method == method for p in programs)
        assert len({p.program_id for p in programs}) == 53
        assert [p.params["support_count"] for p in programs] == sorted(
            (p.params["support_count"] for p in programs), reverse=True
        )
        for program in programs:
            assert sum(fires(program, seq) for seq in db) == program.params["support_count"]
            assert program.score == program.params["support"] == program.params["support_count"] / 4
    limited = mine_prefixspan(db, 0.5, max_pattern_length=2)
    assert pattern_supports(limited) == brute_force_patterns(db, 2, max_items=2)
    assert pattern_supports(mine_cmspade(db, 0.5, max_pattern_length=2)) == pattern_supports(limited)


@requires_spmf
@pytest.mark.parametrize("seed,minsup,max_len", [(0, 0.3, None), (1, 0.5, 2), (2, 0.25, 3), (3, 0.4, 1)])
def test_sequential_miners_match_brute_force(seed, minsup, max_len):
    db = random_db(seed) + [()]  # an empty sequence still counts in the denominator
    min_count = min_support_count(minsup, len(db))
    expected = brute_force_patterns(db, min_count, max_len)
    assert expected
    prefixspan = mine_prefixspan(db, minsup, max_pattern_length=max_len)
    assert pattern_supports(prefixspan) == expected
    assert pattern_supports(mine_cmspade(db, minsup, max_pattern_length=max_len)) == expected
    program = prefixspan[0]
    assert program.params["n_sequences"] == len(db)
    assert program.params["min_support_count"] == min_count
    assert program.params["max_pattern_length"] == max_len
    assert program.params["spmf_version"] == SPMF_VERSION
    tokens = {token for p in prefixspan for itemset in p.elements for token in itemset}
    assert tokens <= set(TOKENS)  # ids were mapped back to tokens


@requires_spmf
@pytest.mark.parametrize("miner,method", [(mine_rulegrowth, "rulegrowth"), (mine_erminer, "erminer")])
def test_documentation_example_rules(miner, method):
    db = doc_db()
    programs = miner(db, 0.75, 0.5)
    got = {
        (tuple(sorted(p.elements[0])), tuple(sorted(p.consequent[0]))): (p.params["support"], p.params["confidence"])
        for p in programs
    }
    assert got == {  # the 9 rules listed on the RuleGrowth documentation page
        (("i1",), ("i2",)): (1.0, 1.0),
        (("i1",), ("i3",)): (1.0, 1.0),
        (("i2",), ("i3",)): (0.75, 0.75),
        (("i3",), ("i2",)): (0.75, 0.75),
        (("i4",), ("i3",)): (0.75, 1.0),
        (("i1", "i3"), ("i2",)): (0.75, 0.75),
        (("i1", "i2"), ("i3",)): (0.75, 0.75),
        (("i1", "i4"), ("i3",)): (0.75, 1.0),
        (("i1",), ("i2", "i3")): (1.0, 1.0),
    }
    assert all(p.kind == "unordered_rule" and p.method == method for p in programs)
    assert all(len(p.elements) == 1 and len(p.consequent) == 1 for p in programs)
    brute = brute_force_rules(db, 3, 0.5)
    assert {(p.elements[0], p.consequent[0]): p.params["support_count"] for p in programs} == {
        key: value[0] for key, value in brute.items()
    }
    limited = miner(db, 0.75, 0.5, max_antecedent=1, max_consequent=1)
    assert len(limited) == 5
    only_antecedent = miner(db, 0.75, 0.5, max_antecedent=1)
    assert {(p.elements[0], p.consequent[0]) for p in only_antecedent} == {
        key for key in brute if len(key[0]) == 1
    }


@requires_spmf
@pytest.mark.parametrize("seed,min_count,minconf", [(10, 2, 0.0), (10, 4, 0.5), (11, 3, 0.3), (12, 3, 0.6)])
def test_rule_miners_against_brute_force(seed, min_count, minconf):
    db = random_db(seed, n=10) + [(), ()]
    minsup = min_count / len(db)
    assert min_support_count(minsup, len(db)) == min_count
    expected = brute_force_rules(db, min_count, minconf)
    assert expected

    erminer = {(p.elements[0], p.consequent[0]): p for p in mine_erminer(db, minsup, minconf)}
    assert set(erminer) == set(expected)  # ERMiner is complete
    for key, (support, confidence) in expected.items():
        assert erminer[key].params["support_count"] == support
        assert erminer[key].params["confidence"] == pytest.approx(confidence, abs=1e-9)

    # SPMF 2.64b RuleGrowth never reports a rule that does not hold or over-counts one,
    # but it can miss rules or under-count them (expandLeft defect, see spmf.py).
    rulegrowth = mine_rulegrowth(db, minsup, minconf)
    for program in rulegrowth:
        key = (program.elements[0], program.consequent[0])
        assert key in expected
        assert program.params["support_count"] <= expected[key][0]


@requires_spmf
def test_rulegrowth_expandleft_defect_minimal_case():
    """<(3 5)(2)>, <(4)>, <(4 5)(3)(2)> at min count 2: {3, 5} ==> {2} holds twice.

    RuleGrowth's expandLeft prunes the new candidate 4 in itemset (4 5) with a jump to
    the next itemset, so 5 is never counted for the third sequence.
    """
    db = [
        (frozenset({"i3", "i5"}), frozenset({"i2"})),
        (frozenset({"i4"}),),
        (frozenset({"i4", "i5"}), frozenset({"i3"}), frozenset({"i2"})),
    ]
    key = (frozenset({"i3", "i5"}), frozenset({"i2"}))
    assert brute_force_rules(db, 2, 0.0)[key][0] == 2
    erminer = {(p.elements[0], p.consequent[0]): p for p in mine_erminer(db, 2 / 3, 0.0)}
    rulegrowth = {(p.elements[0], p.consequent[0]) for p in mine_rulegrowth(db, 2 / 3, 0.0)}
    assert erminer[key].params["support_count"] == 2
    assert key not in rulegrowth
    assert set(erminer) - rulegrowth == {key}


@requires_spmf
def test_erminer_warns_at_min_count_one():
    with pytest.warns(UserWarning, match="ERMiner"):
        mine_erminer(doc_db(), 0.25, 0.5)  # ceil(0.25 * 4) == 1


@requires_spmf
def test_support_denominator_includes_empty_sequences():
    db = doc_db() + [(), (frozenset(),), (), ()]
    programs = mine_prefixspan(db, 0.5)  # min count 4 of 8, i.e. present in all non-empty ones
    assert pattern_supports(programs) == brute_force_patterns(db, 4)
    assert all(p.params["support"] == p.params["support_count"] / 8 for p in programs)
    assert mine_prefixspan(db, 0.75) == []  # 6 > 4 non-empty sequences: SPMF is not called
    assert mine_cmspade([], 0.5) == [] and mine_rulegrowth([(), ()], 0.5, 0.5) == []
    assert mine_erminer([], 0.5, 0.5) == []


@requires_spmf
def test_temporary_files_are_removed(scratch):
    before = set(DEFAULT_TMP_ROOT.iterdir())
    mine_prefixspan(doc_db(), 0.5)
    mine_rulegrowth(doc_db(), 0.75, 0.5)
    mine_erminer(doc_db(), 0.75, 0.5)
    assert set(DEFAULT_TMP_ROOT.iterdir()) == before
    mine_cmspade(doc_db(), 0.5, tmp_root=scratch)
    with pytest.raises(RuntimeError):
        spmf._mine_sequential("NoSuchAlgorithm", "none", doc_db(), 0.5, None, None, 60, scratch)
    assert list(scratch.iterdir()) == []


@requires_spmf
def test_run_spmf_detects_failures(scratch):
    db_path, out_path = scratch / "in.txt", scratch / "out.txt"
    write_spmf_sequence_db(doc_db(), db_path)
    stdout = run_spmf("PrefixSpan", db_path, out_path, ["0.5"])
    assert "PREFIXSPAN" in stdout.upper() and len(parse_sequential_patterns(out_path)) == 53
    # Relative paths are resolved against the caller's cwd, not the JVM's working directory.
    run_spmf("PrefixSpan", os.path.relpath(db_path), os.path.relpath(out_path), ["0.5"])
    assert len(parse_sequential_patterns(out_path)) == 53
    # SPMF exits with code 0 on these errors; run_spmf must still raise.
    with pytest.raises(RuntimeError, match="no algorithm"):
        run_spmf("NoSuchAlgorithm", db_path, out_path, ["0.5"])
    with pytest.raises(RuntimeError, match="incorrect type"):
        run_spmf("PrefixSpan", db_path, out_path, ["0.5", "2", "notabool"])
    assert not out_path.exists()
    with pytest.raises(FileNotFoundError):
        run_spmf("PrefixSpan", db_path, out_path, ["0.5"], jar=scratch / "missing.jar")


@requires_spmf
@pytest.mark.skipif(os.environ.get("SPMF_SWEEP") != "1", reason="slow randomized sweep; set SPMF_SWEEP=1")
def test_randomized_sweep_against_brute_force():
    """150 random databases (about 600 JVM runs, roughly 9 minutes); reproduces the
    numbers in spmf.py. Last run: RuleGrowth affected 22/150 (all with min count >= 2),
    ERMiner at min count 1 affected 1/50."""
    rng = random.Random(7)
    rulegrowth_affected = erminer_min1_affected = 0
    for trial in range(150):
        n = rng.randint(3, 9)
        alphabet = [f"t{i}" for i in range(rng.randint(3, 7))]
        db = [
            tuple(frozenset(rng.sample(alphabet, rng.randint(1, min(4, len(alphabet))))) for _ in range(rng.randint(1, 6)))
            for _ in range(n)
        ]
        min_count = 1 if trial % 3 == 0 else rng.randint(2, max(2, n // 2))
        minsup = min_count / n
        expected_patterns = brute_force_patterns(db, min_count)
        assert pattern_supports(mine_prefixspan(db, minsup)) == expected_patterns
        assert pattern_supports(mine_cmspade(db, minsup)) == expected_patterns
        expected_rules = {key: value[0] for key, value in brute_force_rules(db, min_count, 0.0).items()}
        with pytest.warns(UserWarning) if min_count == 1 else contextlib.nullcontext():
            erminer = {(p.elements[0], p.consequent[0]): p.params["support_count"] for p in mine_erminer(db, minsup, 0.0)}
        if min_count >= 2:
            assert erminer == expected_rules
        elif erminer != expected_rules:
            erminer_min1_affected += 1
        rulegrowth = {(p.elements[0], p.consequent[0]): p.params["support_count"] for p in mine_rulegrowth(db, minsup, 0.0)}
        assert all(key in expected_rules and support <= expected_rules[key] for key, support in rulegrowth.items())
        rulegrowth_affected += rulegrowth != expected_rules
    print(f"RuleGrowth affected: {rulegrowth_affected}/150; ERMiner (min count 1) affected: {erminer_min1_affected}/50")
