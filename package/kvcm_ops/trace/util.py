import json
import subprocess
from pathlib import Path

def _system_grep(file_path, pattern):
    return subprocess.run(
                ['grep', pattern, file_path],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

def system_grep_access_get_all(access_fils, pattern:str) -> list[dict]:
    try:
        results = []
        for file_path in access_fils:
            result = _system_grep(file_path, pattern)
            if result.returncode == 0:
                results += [json.loads(x) for x in result.stdout.strip().split('\n')]
        return results
    except FileNotFoundError as e:
        raise RuntimeError(f"files not exist [{access_fils}] {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON {e}")

def system_grep_access_get_first(access_fils, pattern:str) -> dict:
    try:
        for file_path in access_fils:
            result = _system_grep(file_path, pattern)
            if result.returncode == 0:
                return json.loads(result.stdout.strip().split('\n')[0])
    except FileNotFoundError as e:
        raise RuntimeError(f"files not exist [{access_fils}] {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON {e}")

def list_logs(path:str, log_prefix:str):
    temp_files = []
    for file in Path(path).glob(f'{log_prefix}*'):
        if file.is_file():
            temp_files.append(str(file))
    temp_files.sort()
    files = temp_files[1:] + temp_files[:1]
    return files