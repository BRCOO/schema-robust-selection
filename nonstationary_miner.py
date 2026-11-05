from pathlib import Path

module_path = Path(__file__).with_name("Non-stationary time series.py")
code = module_path.read_text(encoding="utf-8-sig")
exec(compile(code, str(module_path), "exec"), globals(), globals())
