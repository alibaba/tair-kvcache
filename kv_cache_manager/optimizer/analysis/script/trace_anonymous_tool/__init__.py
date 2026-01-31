"""
Trace Anonymous Tool - KVCacheManager Optimizer的标准trace预处理工具链

主要组件：
- tokenizer: Token化处理
- anonymizer: 匿名化处理（token_ids → block_ids）
- utils: 辅助工具（提取、合并、分割）

使用示例：
    from trace_anonymous_tool import tokenizer, anonymizer
    
    # Token化
    tokenizer.gen_token_ids_file(
        file_path="input.jsonl",
        tokenizer=your_tokenizer,
        timestamp_field="time",
        content_field="input"
    )
    
    # 匿名化
    anonymizer.convert_single_file(
        input_file="tokenids_input.jsonl",
        block_size=16,
        truncate=False
    )
"""

__version__ = "1.0.0"
__author__ = "KVCacheManager Optimizer Team"

# 导出主要接口
__all__ = [
    'tokenizer',
    'anonymizer',
    'utils',
]
