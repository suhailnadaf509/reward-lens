"""MkDocs build hooks: put the generated catalogue into the site, and stop the build if it lies.

A native MkDocs hook rather than a plugin, so the docs build needs no dependency it did not
already have. `on_files` is the right event: it runs before the nav is built, so a generated page
can be named in `nav:` exactly like a hand-written one, and `--strict` will fail loudly if one goes
missing rather than shipping a site with a hole in it.

The pages are generated into memory and never written to `docs/content`. That is deliberate in
three ways. There is no committed copy to go stale. A review diff does not carry ninety
machine-written files. And `reward-lens-claims`, which reads `docs/content` off disk, keeps its
ratchet over the prose a person wrote, which is the prose it exists to police.

If `reward_lens` will not import, this hook fails the build. It does not fall back to an empty
catalogue: a docs build that silently skips the registry leaves lint rule two unenforced while
still looking green, and an unenforced gate that reports success is worse than a red build. The
documentation environment has to install the package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_catalogue  # noqa: E402


def on_files(files: Any, config: Any, **_: Any) -> Any:
    from mkdocs.exceptions import PluginError
    from mkdocs.structure.files import File

    try:
        pages, notes = gen_catalogue.generate(check=True)
    except gen_catalogue.LintFailure as exc:
        raise PluginError(str(exc)) from exc

    for note in notes:
        print(note)
    for uri, text in pages.items():
        files.append(File.generated(config, uri, content=text))
    print(f"catalogue: {len(pages)} pages rendered from the registry")
    return files


def on_page_context(context: Any, page: Any, **_: Any) -> Any:
    """Generated pages have no file to edit, so do not offer a pencil that leads nowhere."""
    if page.file.abs_src_path is None:
        page.edit_url = None
    return context
