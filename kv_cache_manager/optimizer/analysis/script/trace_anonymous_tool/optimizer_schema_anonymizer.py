# ============================================================
# File Name: optimizer_schema_anonymizer.py
# Description:
#   - This script converts token id sequences into Optimizer-compatible
#     DialogTurnSchemaTrace JSON format with prefix-dependent block ids.
#
# Input:
#   - tokenids_*.jsonl file containing 'token_ids' field
#
# Output:
#   - optimizer_trace_*.jsonl file in DialogTurnSchemaTrace JSON format
#     containing all required fields for Optimizer
# ============================================================
import os
import json
import argparse
import tqdm
import hashlib
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANONYMIZED_DIR = os.path.join(SCRIPT_DIR, 'result/anonymous_files')

class BlockIdAnonymizer:
    def __init__(self, block_size: int=16, enable_truncate: bool=False, instance_id: str='instance', block_mask_mode: str='empty_array'):
        self._hash_to_id = {}
        self._id_counter = 0
        self.block_size = block_size
        self.instance_id = instance_id
        self.enable_truncate = enable_truncate
        self.block_mask_mode = block_mask_mode

    def tokens_to_block_ids(self, token_ids: list[int], prev_hash: str = 'prefix_start') -> list[int]:
        block_ids = []
        for i in range(0, len(token_ids), self.block_size):
            if self.enable_truncate and i + self.block_size > len(token_ids):
                continue
            block_tokens = token_ids[i : i + self.block_size]
            key_tuple = (prev_hash, tuple(block_tokens))
            block_hash = hashlib.md5(str(key_tuple).encode()).hexdigest()
            anonymized_id = self._get_or_create_block_id(block_hash)
            block_ids.append(anonymized_id)
            prev_hash = block_hash
        return block_ids

    def _get_or_create_block_id(self, hash_key):
        if hash_key not in self._hash_to_id:
            self._id_counter += 1
            self._hash_to_id[hash_key] = self._id_counter
        return self._hash_to_id[hash_key]

    def _generate_block_mask(self, block_ids: list[int]):
        """生成 block_mask
        
        Args:
            block_ids: block ID 列表
        
        Returns:
            根据 block_mask_mode 返回：
            - 'empty_array': 空数组 []
            - 'offset': 整数偏移量 0
            - 'full': 全 True 数组 [True, True, ...]
        """
        if self.block_mask_mode == 'empty_array':
            return []
        elif self.block_mask_mode == 'offset':
            return 0
        elif self.block_mask_mode == 'full':
            return [True] * len(block_ids)
        else:
            return []  # 默认返回空数组

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._hash_to_id, f)

def convert_single_file(input_file: str, block_size: int, truncate: bool, instance_id: str = 'instance', block_mask_mode: str = 'empty_array'):
    anonymizer = BlockIdAnonymizer(block_size=block_size, enable_truncate=truncate, instance_id=instance_id, block_mask_mode=block_mask_mode)
    basename = os.path.basename(input_file)
    out_file = os.path.join(ANONYMIZED_DIR, basename.replace('tokenids_', 'optimizer_trace_'))
    
    os.makedirs(ANONYMIZED_DIR, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm.tqdm(fin, desc=f"Converting to Optimizer format: {basename}"):
            data = json.loads(line)
            block_ids = anonymizer.tokens_to_block_ids(data['token_ids'])
            
            # OptimizerSchemaTrace 格式（DialogTurnSchemaTrace）
            result = {
                # 基础字段（OptimizerSchemaTrace）
                'instance_id': instance_id,
                'trace_id': f"trace_{instance_id}_{int(data['timestamp'] * 1000000)}",
                'timestamp_us': int(data['timestamp'] * 1000000),
                'tokens': [],
                'keys': block_ids,
                
                # GetLocationSchemaTrace 字段
                'query_type': 'prefix_match',
                'block_mask': anonymizer._generate_block_mask(block_ids),
                'sw_size': 0,
                'location_spec_names': [],
                
                # DialogTurnSchemaTrace 字段
                'input_len': len(block_ids) * block_size,
                'output_len': 0,
                'total_keys': block_ids
            }
            
            fout.write(json.dumps(result) + '\n')
    
    print(f"生成了 {anonymizer._id_counter} 个唯一 block")
    print(f"输出文件: {out_file}")

if __name__ == '__main__':
    """ python optimizer_schema_anonymizer.py --file_path <input_file> --block_size <block_size> --instance-id <instance_id> --block-mask-mode <mode> """
    parser = argparse.ArgumentParser(description="将 token ids 转换为 Optimizer 标准格式的 trace")
    parser.add_argument('--file_path', required=True, help='输入文件路径（tokenids_*.jsonl）')
    parser.add_argument('--block_size', default=16, type=int, help='Block 大小（默认: 16）')
    parser.add_argument('--truncate', default=False, action='store_true', help='是否截断不完整的 block')
    parser.add_argument('--instance-id', default='instance', type=str, help='实例 ID（默认: instance）')
    parser.add_argument('--block-mask-mode', default='empty_array', 
                       choices=['empty_array', 'offset', 'full'],
                       help='Block mask 模式：empty_array=空数组[], offset=整数偏移量0, full=全True数组（默认: empty_array）')
    
    args = parser.parse_args()
    
    convert_single_file(args.file_path, args.block_size, args.truncate, args.instance_id, args.block_mask_mode)