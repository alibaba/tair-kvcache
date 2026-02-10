import os
from hisim.utils.logger import get_logger

logger = get_logger("hisim")


class Envs:
    @classmethod
    def config_path(cls) -> str:
        HISIM_CONFIG_PATH = os.getenv("HISIM_CONFIG_PATH")
        if not HISIM_CONFIG_PATH or not os.path.exists(HISIM_CONFIG_PATH):
            raise RuntimeError(
                f"The mock configuration path is not set or does not exist({HISIM_CONFIG_PATH}). Please set it using the system variable HISIM_CONFIG_PATH"
            )
        return HISIM_CONFIG_PATH

    @classmethod
    def output_dir(cls) -> str:
        HISIM_OUTPUT_DIR = os.getenv("HISIM_OUTPUT_DIR", "/tmp/hisim/output/")
        HISIM_OUTPUT_DIR = os.path.realpath(HISIM_OUTPUT_DIR)
        if os.path.exists(HISIM_OUTPUT_DIR) and os.path.isfile(HISIM_OUTPUT_DIR):
            logger.error(
                f"The metrics output path, {HISIM_OUTPUT_DIR}, exists and is a file."
            )
            raise RuntimeError(f"{HISIM_OUTPUT_DIR} exists but is not a directory.")
        os.makedirs(os.path.dirname(HISIM_OUTPUT_DIR), exist_ok=True)
        return HISIM_OUTPUT_DIR

    @classmethod
    def simulation_mode(cls) -> str:
        HISIM_SIMULATION_MODE = os.getenv("HISIM_SIMULATION_MODE", "OFFLINE").upper()
        assert HISIM_SIMULATION_MODE in ("BLOCKING", "OFFLINE")
        return HISIM_SIMULATION_MODE

    @classmethod
    def num_warmup(cls) -> int:
        # The number of warmup requests.
        return int(os.getenv("HISIM_NUM_WARMUP", "0"))

    @classmethod
    def reset_hicache_storage(cls) -> bool:
        return os.getenv("HISIM_RESET_HICACHE_STORAGE") == "1"
