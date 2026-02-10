from pathlib import Path
from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo

from hisim.time_predictor import (
    AIConfiguratorTimePredictor,
    ScheduleBatch,
    FakeRequest,
)
from hisim.simulation.types import SchedulerConfig


cur_dir = Path(__file__).parent


def test_aic_xgb_predictor():
    model = ModelInfo.from_modelscope_id("Qwen/Qwen3-8B")
    hw = AcceleratorInfo.find_by_hw_name("H20")
    hw.name = "h100_sxm"  # AIConfigurator internal device name
    config = SchedulerConfig(
        model=model, backend_name="sglang", backend_version="0.5.6.post2"
    )

    xgb_model_path = cur_dir / "assets/aic_xgb_models/"
    predictor = AIConfiguratorTimePredictor(
        model, hw, config, xgb_model_path=xgb_model_path
    )

    # Prefill
    reqs = [
        FakeRequest(512, 512),
        FakeRequest(1024, 0),
        FakeRequest(512, 0),
    ]

    latency = predictor.predict_infer_time(ScheduleBatch(reqs))
    assert latency > 0

    # Decode
    reqs = [
        FakeRequest(1, 1024),
        FakeRequest(1, 1024),
        FakeRequest(1, 1024),
    ]

    latency = predictor.predict_infer_time(ScheduleBatch(reqs))
    assert latency > 0


if __name__ == "__main__":
    test_aic_xgb_predictor()
