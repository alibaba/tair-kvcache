import os
import signal
import subprocess
import sys
import requests
import json
import time
import pytest

from hisim.simulation.base.runner import BaseBenchmarkRunner
from env import check_framework, MODEL_PATH

os.environ["HISIM_CONFIG_PATH"] = os.path.dirname(__file__) + "/assets/mock/config.json"


class SGLangServingRunner(BaseBenchmarkRunner):
    def __init__(self, server_args: dict):

        cmd = [sys.executable, "-m", "hisim.simulation.sglang.launch_server"]

        for k, v in server_args.items():
            flag = "--" + k.replace("_", "-")
            if v is True:
                cmd.append(flag)
            elif v is False:
                pass
            else:
                cmd.extend([flag, str(v)])

        self.server_proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

        dur = 0
        while dur < 120:
            try:
                r = requests.get(url="http://localhost:30000")
                if r.status_code < 500:
                    return
            except Exception:
                pass
            time.sleep(1)
            dur += 1
        raise RuntimeError("Fail to start llm server.")

    def benchmark(self):
        start = time.time()
        output_file = "/tmp/hisim_serving_benchmark.json"
        cmd = [
            sys.executable,
            "-m",
            "hisim.simulation.bench_serving",
            "--warmup-request=0",
            "--bench-mode=simulation",
            "--backend=sglang",
            "--dataset-name=sharegpt",
            "--num-prompts=10",
            f"--output-file={output_file}",
        ]
        subprocess.run(cmd)

        with open(output_file) as f:
            metrics = json.load(f)

        with open(output_file, "w") as f:
            # clear data
            pass

        metrics["time_cost"] = time.time() - start

        return metrics

    def flush_cache(self):
        requests.get(url="http://localhost:30000/flush_cache")

    def shutdown(self):
        if not self.server_proc or self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()
        self.server_proc = None


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
def test_benchmark():
    runner = SGLangServingRunner(
        server_args={"model_path": MODEL_PATH, "device": "cpu"}
    )

    metrics = runner.benchmark()

    runner.shutdown()
    assert metrics["completed"] == 10


if __name__ == "__main__":
    test_benchmark()
