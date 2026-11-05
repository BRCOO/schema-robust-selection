"""Print the definition of a registry program and its surrogate chain (for the resolution figure of the method section)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e3_context", ROOT / "scripts/v2/e3_context.py")
ctx = importlib.util.module_from_spec(spec)
sys.modules["e3_context"] = ctx
spec.loader.exec_module(ctx)

run = ctx.load_run(*ctx.run_dirs("union", "f_mimic_iv_gap1h"))
comp = run.registries["srs_ap_t"]
pid = sys.argv[1] if len(sys.argv) > 1 else "erminer_016775"
p = next(x for x in comp.primaries if x.program_id == pid)
for b in comp.branches(p):
    print(b.program_id, b.kind, "elements=", [sorted(e) for e in b.elements], "consequent=", [sorted(e) for e in b.consequent],
          "features=", sorted(b.features))
for t in run.targets:
    print(t.name, sorted(set(run.source) - set(t.features)))
