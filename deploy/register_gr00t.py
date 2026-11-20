#!/usr/bin/env python3
"""Register ``--policy gr00t`` in a samsung_deploy checkout: copies ``gr00t.py`` next to the other
policies and adds one loader + one registry entry to ``vla_deploy/policy/__init__.py``.  Nothing else
is touched; running it twice changes nothing.

    python register_gr00t.py /coc/flash2/jpark984/samsung_deploy
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

LOADER = '''def _gr00t(**kwargs: Any) -> VlaPolicy:
    from .gr00t import Gr00tHalabPolicy

    return Gr00tHalabPolicy(**kwargs)


# name -> (loader, one-line description, required --policy-arg keys)
'''
ENTRY = '''    "gr00t": (
        _gr00t,
        "GR00T N1.7 HA-Lab baseline (47-D absolute, 32 rows held to chunk_len) served by Isaac-GR00T "
        "examples/HALAB/halab_gr00t_server.py; --policy-arg server=tcp://host:port [chunk_len=40] [pitch_scale=2.0] "
        "[splice_blend_rows=4] [head_max_step=0.10] [start_blend_rows=15] [arm_offset_comp=none|command]",
        ("server",),
    ),
    # Add baselines here. Each entry is one lazy loader; see skeleton.py.
'''


def main() -> int:
    root = Path(sys.argv[1]) / "vla_deploy" / "policy"
    init = root / "__init__.py"
    text = init.read_text()
    shutil.copyfile(Path(__file__).with_name("gr00t.py"), root / "gr00t.py")
    if '"gr00t": (' in text:
        print(f"{init}: already registered; gr00t.py updated")
        return 0
    anchors = ("# name -> (loader, one-line description, required --policy-arg keys)\n",
               "    # Add baselines here. Each entry is one lazy loader; see skeleton.py.\n")
    for anchor in anchors:
        if text.count(anchor) != 1:
            raise SystemExit(f"{init}: expected exactly one line {anchor.strip()!r}; register by hand")
    text = text.replace(anchors[0], LOADER).replace(anchors[1], ENTRY)
    init.write_text(text)
    print(f"{init}: gr00t registered; {root / 'gr00t.py'} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
