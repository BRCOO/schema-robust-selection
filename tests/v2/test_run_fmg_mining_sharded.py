from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "v2" / "run_fmg_mining_sharded.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_fmg_mining_sharded", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shard_metas_is_balanced_and_order_independent():
    m = _module()
    metas = [{"episode_id": f"stay_{i}_positive"} for i in range(10)]
    a = m.shard_metas(metas, 3)
    b = m.shard_metas(list(reversed(metas)), 3)
    assert a == b
    assert sorted(len(s) for s in a) == [3, 3, 4]
    assert sorted(e["episode_id"] for s in a for e in s) == sorted(e["episode_id"] for e in metas)
    assert len(m.shard_metas(metas[:2], 8)) == 2
    with pytest.raises(ValueError):
        m.shard_metas(metas, 0)


def test_merge_shard_caches_streams_and_rejects_duplicates(tmp_path):
    m = _module()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps([{"episode_id": "e1", "rules": [{"rule_id": "r"}]}], indent=2), encoding="utf-8")
    b.write_text(json.dumps([{"episode_id": "e2", "rules": []}, {"episode_id": "e3", "rules": []}]), encoding="utf-8")
    merged = tmp_path / "merged.json"
    out = m.merge_shard_caches([a, b, tmp_path / "missing.json"], merged)
    assert out["n_episodes"] == 3
    assert [item["episode_id"] for item in json.loads(merged.read_text(encoding="utf-8"))] == ["e1", "e2", "e3"]
    b.write_text(json.dumps([{"episode_id": "e1", "rules": []}]), encoding="utf-8")
    with pytest.raises(ValueError, match="more than one shard"):
        m.merge_shard_caches([a, b], tmp_path / "dup.json")
    assert not (tmp_path / "dup.json").exists()


def test_merge_can_keep_only_selected_rule_ids(tmp_path):
    m = _module()
    a = tmp_path / "a.json"
    a.write_text(json.dumps([{"episode_id": "e1", "rules": [{"rule_id": "r1"}, {"rule_id": "r2"}]}, {"episode_id": "e2", "rules": [{"rule_id": "r3"}]}]), encoding="utf-8")
    merged = tmp_path / "m.json"
    m.merge_shard_caches([a], merged, frozenset({"r2"}))
    items = json.loads(merged.read_text(encoding="utf-8"))
    assert [i["episode_id"] for i in items] == ["e1", "e2"]
    assert items[0]["rules"] == [{"rule_id": "r2"}] and items[1]["rules"] == []
