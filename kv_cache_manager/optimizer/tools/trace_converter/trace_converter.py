#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace Converter - 将各种trace格式转换为Optimizer标准格式

Usage:
    # Publisher日志转换
    python trace_converter.py -i publisher.log -o standard.jsonl -f publisher_log --mode optimizer

    # Qwen Bailian数据集转换
    python trace_converter.py -i qwen_trace.jsonl -o standard.jsonl -f qwen_bailian --mode optimizer

    # 文本对话转换
    python trace_converter.py -i text.jsonl -o standard.jsonl -f text --mode optimizer \
        --tokenizer-path model/deepseek-v3 --block-size 16
"""

import argparse
import sys
import os
from pathlib import Path

# 添加模块路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from converters.publisher_log import PublisherLogConverter
from converters.qwen_bailian import QwenBailianConverter
from converters.text_trace import TextTraceConverter


# ========================================
# Converter 注册表
# ========================================
CONVERTERS = {
    'publisher_log': PublisherLogConverter,
    'qwen_bailian': QwenBailianConverter,
    'text': TextTraceConverter,
}

# ========================================
# 扩展 Converter（可选）
# ========================================
# 如需添加自定义 converter，在此处动态注册：
#
# try:
#     from converters.custom import CustomConverter
#     CONVERTERS['custom'] = CustomConverter
# except ImportError:
#     pass  # 如果模块不存在，静默忽略
#
# 所有 converter 会自动获得全部参数：
#   - default_instance_id, instance_block_sizes, mode
#   - tokenizer_path, model_name, model_mapping
#   - time_field, content_field, num_workers
# 在 __init__ 中用 **kwargs 忽略不需要的参数


def main():
    available_formats = list(CONVERTERS.keys())
    
    parser = argparse.ArgumentParser(
        description='Convert various trace formats to Optimizer standard format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 必需参数
    parser.add_argument('-i', '--input', required=True,
                        help='Input trace file path')
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Output standard trace file path (default: auto-generate based on input filename and mode)')
    parser.add_argument('-f', '--format', required=True,
                        choices=available_formats,
                        help=f'Input trace format (available: {", ".join(available_formats)})')

    # 输出模式
    parser.add_argument('--mode', default='optimizer',
                        choices=['optimizer', 'inference'],
                        help='Output mode: optimizer (Get+Write) or inference (DialogTurn). Default: optimizer')

    # 通用可选参数
    parser.add_argument('--instance-id', default='instance',
                        help='Default instance ID (default: instance)')
    parser.add_argument(
        '--instance-block-sizes',
        default=None,
        help='Per-instance block sizes (format: instance1:16,instance2:32). Unspecified instances use default 16')

    # Tokenizer相关参数 (text格式需要)
    parser.add_argument('--tokenizer-path', default=None,
                        help='Tokenizer path (local path or HuggingFace model ID)')
    parser.add_argument(
        '--model-name',
        default=None,
        help='Model name (if specified and ./model/{model_name} exists, use it instead of tokenizer-path)')
    parser.add_argument(
        '--model-mapping',
        default=None,
        help='Model name mapping (format: "GLM-4.6:zai-org/GLM-4.6,Qwen2.5:Qwen/Qwen2.5-7B-Instruct")')

    # text格式特有参数
    parser.add_argument('--time-field', default='time',
                        help='Time field name for text format (default: time)')
    parser.add_argument('--content-field', default='prompt_messages',
                        help='Content field name for text format (default: prompt_messages)')
    
    # 性能优化参数
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers for tokenization (default: CPU count - 1)')

    args = parser.parse_args()

    # 验证输入文件存在
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # 自动生成输出文件名 (如果未指定)
    if args.output is None:
        # 获取输入文件的目录和文件名(不含扩展名)
        input_dir = input_path.parent
        input_stem = input_path.stem  # 文件名不含扩展名

        # 根据mode生成后缀
        if args.mode == 'optimizer':
            suffix = '_optimizer'
        else:  # inference
            suffix = '_inference'

        # 生成输出文件名
        output_filename = f"{input_stem}{suffix}.jsonl"
        output_path = input_dir / output_filename

        print(f"📝 Auto-generated output file: {output_path}")
    else:
        output_path = Path(args.output)

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 解析instance-block-sizes参数
    instance_block_sizes = {}
    if args.instance_block_sizes:
        try:
            for pair in args.instance_block_sizes.split(','):
                inst_id, size = pair.split(':')
                instance_block_sizes[inst_id.strip()] = int(size.strip())
        except ValueError as e:
            print(f"❌ Error: Invalid --instance-block-sizes format. Expected 'instance1:16,instance2:32'", file=sys.stderr)
            return 1

    # 解析model-mapping参数
    model_mapping = None
    if args.model_mapping:
        model_mapping = {}
        try:
            for pair in args.model_mapping.split(','):
                src_model, hf_model = pair.split(':')
                model_mapping[src_model.strip()] = hf_model.strip()
        except ValueError as e:
            print(
                f"❌ Error: Invalid --model-mapping format. Expected 'GLM-4.6:THUDM/glm-4-9b-chat,Qwen:Qwen/Qwen2.5-7B-Instruct'",
                file=sys.stderr)
            return 1

    # 选择转换器
    try:
        converter_class = CONVERTERS.get(args.format)
        if not converter_class:
            print(f"❌ Error: Unsupported format: {args.format}", file=sys.stderr)
            print(f"   Available formats: {', '.join(available_formats)}", file=sys.stderr)
            return 1
        
        # 统一参数传递（所有 converter 获得全部参数）
        # 每个 converter 在 __init__ 中选择性接收需要的参数
        converter = converter_class(
            default_instance_id=args.instance_id,
            instance_block_sizes=instance_block_sizes,
            mode=args.mode,
            tokenizer_path=args.tokenizer_path,
            model_name=args.model_name,
            model_mapping=model_mapping,
            time_field=args.time_field,
            content_field=args.content_field,
            num_workers=args.num_workers
        )

        # 执行转换
        print(f"🔄 Converting {args.input} → {output_path}")
        print(f"   Format: {args.format}")
        print(f"   Mode: {args.mode}")
        print(f"   Default Instance ID: {args.instance_id}")

        trace_count = converter.convert(args.input, str(output_path))

        print(f"✅ Success! Converted {trace_count} traces")
        print(f"   Output: {output_path}")
        return 0

    except Exception as e:
        print(f"❌ Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
