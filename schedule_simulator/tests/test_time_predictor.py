from kunlun_commons.model_info import ModelInfo
from kunlun_commons.hardwares.accelerator import AcceleratorInfo

from schedule_simulator.schedule_emulator.types import SchedulerConfig
from schedule_simulator.infer_time_predictor import (
    LLMPerfTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)


def _make_predictor(model_name="Qwen2.5-3B", device="H20", tp=1):
    model = ModelInfo.find_by_model_name(model_name)
    hw = AcceleratorInfo.find_by_hw_name(device)
    config = SchedulerConfig(model=model, tp_size=tp)
    return LLMPerfTimePredictor(model=model, hw=hw, config=config)


def test_llmperf_predictor_prefill():
    predictor = _make_predictor()
    batch = ScheduleBatch(
        reqs=[ScheduleRequest(input_length=512, past_kv_length=0)],
    )
    latency = predictor.predict_infer_time(batch)
    assert latency > 0, f"prefill latency should be positive, got {latency}"
    assert batch.is_prefill()
    print(f"Prefill (512 tok): {latency*1000:.2f} ms")


def test_llmperf_predictor_decode():
    predictor = _make_predictor()
    batch = ScheduleBatch(
        reqs=[
            ScheduleRequest(input_length=1, past_kv_length=512),
            ScheduleRequest(input_length=1, past_kv_length=256),
            ScheduleRequest(input_length=1, past_kv_length=1024),
        ],
    )
    latency = predictor.predict_infer_time(batch)
    assert latency > 0, f"decode latency should be positive, got {latency}"
    assert batch.is_decode()
    assert batch.batch_size == 3
    print(f"Decode (bs=3): {latency*1000:.2f} ms")


def test_llmperf_predictor_mixed():
    predictor = _make_predictor()
    batch = ScheduleBatch(
        reqs=[
            ScheduleRequest(input_length=256, past_kv_length=0),
            ScheduleRequest(input_length=1, past_kv_length=512),
            ScheduleRequest(input_length=1, past_kv_length=300),
        ],
    )
    latency = predictor.predict_infer_time(batch)
    assert latency > 0, f"mixed latency should be positive, got {latency}"
    assert batch.is_prefill()
    print(f"Mixed (1 prefill + 2 decode): {latency*1000:.2f} ms")


def test_predictor_different_models():
    for model_name in ["Qwen2.5-3B", "Qwen2.5-7B"]:
        predictor = _make_predictor(model_name=model_name)
        batch = ScheduleBatch(
            reqs=[ScheduleRequest(input_length=1, past_kv_length=256)],
        )
        latency = predictor.predict_infer_time(batch)
        assert latency > 0
        print(f"{model_name} decode: {latency*1000:.2f} ms")


def test_latency_scaling():
    """Verify larger batch has higher or equal latency than smaller batch."""
    predictor = _make_predictor()
    small_batch = ScheduleBatch(
        reqs=[ScheduleRequest(input_length=1, past_kv_length=256)],
    )
    large_batch = ScheduleBatch(
        reqs=[ScheduleRequest(input_length=1, past_kv_length=256) for _ in range(8)],
    )
    small_lat = predictor.predict_infer_time(small_batch)
    large_lat = predictor.predict_infer_time(large_batch)
    assert large_lat >= small_lat, f"larger batch should not be faster: {large_lat} < {small_lat}"
    print(f"Scaling: bs=1 {small_lat*1000:.2f}ms, bs=8 {large_lat*1000:.2f}ms")


if __name__ == "__main__":
    test_llmperf_predictor_prefill()
    test_llmperf_predictor_decode()
    test_llmperf_predictor_mixed()
    test_predictor_different_models()
    test_latency_scaling()
