#!/usr/bin/env python
"""Regenerate the JSON twins of `spec/*.yaml`.

The YAML is the edited source and carries the comments saying where each row came from. The JSON is
what the registry loads and what the wheel ships, because the core installs with no compiled
dependency and PyYAML has one. Two copies of the JSON exist and both must be written: `spec/` beside
the YAML, and `src/reward_lens/spec/` inside the package.

`tests/acceptance/test_w1_kernel.py::test_the_packaged_json_catalogue_agrees_with_the_yaml_source`
is what keeps the three files the same file. Run this after any edit to the YAML, then run that
test. Needs PyYAML, which is a dev dependency rather than a runtime one.

    python tools/regen_spec_json.py
"""

from __future__ import annotations

import json
import pathlib
import sys

STEMS = ("CATALOGUE", "QUANTITIES")


def main() -> int:
    try:
        import yaml
    except ModuleNotFoundError:
        print("PyYAML is not installed. It is a dev dependency: pip install -e '.[dev]'")
        return 1

    root = pathlib.Path(__file__).resolve().parents[1]
    for stem in STEMS:
        src = root / "spec" / f"{stem}.yaml"
        if not src.exists():
            print(f"{src} is missing; this is not a source checkout")
            return 1
        data = yaml.safe_load(src.read_text(encoding="utf-8"))
        top = next(iter(data))
        rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        for dst in (
            root / "spec" / f"{stem}.json",
            root / "src" / "reward_lens" / "spec" / f"{stem}.json",
        ):
            dst.write_text(rendered, encoding="utf-8")
        print(f"{stem}: {len(data[top])} rows written to both copies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
