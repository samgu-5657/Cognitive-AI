"""Migration test: a pre-Phase-2 database must open read/write safely.

Uses a COPY of the real pre-audit backup in tmp (never the live file).
Skipped when the backup is absent (e.g. fresh machines).
"""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitive_ai.memory import Memory

BACKUP = (r"C:\Users\samgu\AppData\Local\Temp\opencode"
          r"\cognitive.db.pre-audit-bak")


def test_old_database_migrates_compatibly(tmp_path):
    if not os.path.exists(BACKUP):
        pytest.skip("pre-audit backup not present on this machine")
    dst = str(tmp_path / "old_copy.db")
    shutil.copy(BACKUP, dst)
    mem = Memory(dst)  # must not raise; migration is additive
    try:
        concepts = mem.list_concepts()
        assert any(c["id"] == "cup" for c in concepts)
        cup_mats = [a["value"] for a in mem.get_attributes("cup")
                    if a["key"] == "material"]
        assert "ceramic" in cup_mats  # old verified knowledge preserved
        # new structures usable
        eid = mem.record_lifecycle("attribute", "cup.material",
                                   "verified", "contested",
                                   "migration check", [])
        assert eid
        assert mem.list_lifecycle("cup.material")
        evs = mem.list_evidences()
        assert isinstance(evs, list)  # new columns default, old rows readable
        assert mem.current_version() >= 0
    finally:
        mem.close()
