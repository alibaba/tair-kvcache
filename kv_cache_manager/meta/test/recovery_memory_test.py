"""Opt-in allocator regression; each scenario uses a fresh allocator process."""

import json
import os
import subprocess
import unittest


class RecoveryMemoryTest(unittest.TestCase):
    def run_comparison(self):
        library = os.environ.get("JEMALLOC_LIBRARY")
        self.assertTrue(library and os.path.isabs(library) and os.path.isfile(library),
                        "set JEMALLOC_LIBRARY to an absolute libjemalloc.so path")
        probe = os.path.join(os.path.dirname(__file__), "recovery_memory_probe")
        results = {}
        for enabled in (False, True):
            env = dict(os.environ)
            env.update(
                LD_PRELOAD=library,
                MALLOC_CONF="narenas:4,percpu_arena:disabled,background_thread:false,dirty_decay_ms:10000,muzzy_decay_ms:0",
                KVCM_RECOVER_ARENA_ROTATION_ENABLED=str(enabled).lower(),
            )
            process = subprocess.run([probe], env=env, text=True, capture_output=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            records = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
            print(json.dumps({"rotation": enabled, "records": records}), flush=True)
            samples = {row["phase"]: row for row in records if "phase" in row}
            self.assertEqual(set(samples), {"recovered", "updated_1", "updated_2"})
            recovered = samples["recovered"]
            for sample in samples.values():
                self.assertEqual(sample["keys"], 262144)
                self.assertEqual(sample["charge"], recovered["charge"])
                self.assertGreaterEqual(sample["active"], sample["allocated"])
                self.assertLess(abs(sample["allocated"] - recovered["allocated"]), recovered["allocated"] * 0.05)
            if enabled:
                for arena in recovered["arenas"]:
                    self.assertGreater(arena["allocated"], recovered["allocated"] * 0.15)
                    self.assertLess(arena["allocated"], recovered["allocated"] * 0.35)
            else:
                self.assertGreater(recovered["arenas"][0]["allocated"], recovered["allocated"] * 0.90)
            results[enabled] = samples
        self.assertEqual(results[False]["recovered"]["charge"], results[True]["recovered"]["charge"])
        return results[False], results[True]

    @staticmethod
    def gap(sample):
        return sample["active"] - sample["allocated"]

    def test_updates_cover_all_recovery_arenas(self):
        off, on = self.run_comparison()
        # Establish that the negative control actually reproduced inflation.
        self.assertGreater(self.gap(off["updated_2"]), off["recovered"]["allocated"] * 0.10)
        for phase in ("updated_1", "updated_2"):
            self.assertLess(self.gap(on[phase]), self.gap(off[phase]) * 0.80)
        # Guard against continuing growth after the first update sweep.
        self.assertLess(on["updated_2"]["active"] - on["updated_1"]["active"], on["recovered"]["allocated"] * 0.05)


if __name__ == "__main__":
    unittest.main()
