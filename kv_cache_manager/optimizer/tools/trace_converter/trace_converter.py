#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace Converter - 将各种trace格式转换为Optimizer标准格式

Usage:
    python trace_converter.py -i publisher.log -o standard.jsonl -f publisher_log
    python trace_converter.py -i qwen_trace.jsonl -o standard.jsonl -f qwen_bailian
    python trace_converter.py -i text.jsonl -o standard.jsonl -f text --tokenizer-path model/deepseek-v3
    python trace_converter.py -i input.jsonl -o output.jsonl -f custom --converter-module /path/to/custom.py
"""

import argparse
import sys
import os
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Type, List, Optional

# 添加模块路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from converters.base import BaseConverter


def _camel_to_snake(name: str) -> str:
    """驼峰命名转下划线命名"""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            # 检查前一个字符是否为小写，或下一个字符是否为小写
            if name[i-1].islower() or (i < len(name)-1 and name[i+1].islower()):
                result.append('_')
        result.append(char.lower())
    return ''.join(result)


def _infer_format_name(class_name: str) -> str:
    """从类名推断format名称（去掉Converter后缀并转下划线）"""
    if class_name.endswith('Converter'):
        base_name = class_name[:-9]
        return _camel_to_snake(base_name)
    return _camel_to_snake(class_name)


def _load_module_from_file(file_path: Path) -> Optional[object]:
    """从文件路径动态加载Python模块"""
    try:
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[file_path.stem] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"⚠️  Warning: Failed to load module {file_path}: {e}", file=sys.stderr)
        return None


def _discover_converters_in_dir(scan_dir: Path) -> Dict[str, Type[BaseConverter]]:
    """扫描目录发现所有converter类"""
    converters = {}
    
    if not scan_dir.exists() or not scan_dir.is_dir():
        return converters
    
    # 递归扫描所有.py文件
    for py_file in scan_dir.rglob('*.py'):
        # 跳过特殊文件
        if py_file.name in ('__init__.py', 'base.py'):
            continue
        
        # 动态加载模块
        module = _load_module_from_file(py_file)
        if module is None:
            continue
        
        # 查找所有继承BaseConverter的类
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # 必须继承BaseConverter且不是BaseConverter自身
            if issubclass(obj, BaseConverter) and obj is not BaseConverter:
                format_name = _infer_format_name(name)
                converters[format_name] = obj
                print(f"✓ Discovered converter: '{format_name}' ({name} from {py_file.name})")
    
    return converters


def _register_converter_from_file(file_path: str) -> Dict[str, Type[BaseConverter]]:
    """从文件路径注册converter"""
    converters = {}
    
    try:
        path_obj = Path(file_path)
        
        # 判断是文件路径还是模块路径
        if path_obj.exists() and path_obj.suffix == '.py':
            # 文件路径: 动态加载
            module = _load_module_from_file(path_obj)
            if module is None:
                return converters
        else:
            # 模块路径: 使用importlib.import_module
            try:
                module = importlib.import_module(file_path)
            except ImportError as e:
                print(f"❌ Error: Failed to import module '{file_path}': {e}", file=sys.stderr)
                return converters
        
        # 自动查找所有继承BaseConverter的类
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseConverter) and obj is not BaseConverter:
                format_name = _infer_format_name(name)
                converters[format_name] = obj
                print(f"✓ Registered converter: '{format_name}' ({name} from {file_path})")
        
    except Exception as e:
        print(f"❌ Error: Failed to register converter from '{file_path}': {e}", file=sys.stderr)
    
    return converters


def _build_converter_registry(
    builtin_dir: Path,
    extra_dirs: List[Path],
    module_specs: List[str]
) -> Dict[str, Type[BaseConverter]]:
    """构建完整的converter注册表"""
    converters = {}
    
    print("🔍 Scanning converters...")
    builtin_converters = _discover_converters_in_dir(builtin_dir)
    converters.update(builtin_converters)
    
    for extra_dir in extra_dirs:
        is_auto_discovered = 'stub_source' in str(extra_dir)
        
        if not is_auto_discovered:
            print(f"🔍 Scanning directory: {extra_dir}")
        
        extra_converters = _discover_converters_in_dir(extra_dir)
        for name, cls in extra_converters.items():
            if name in converters and not is_auto_discovered:
                print(f"  ⚠️  Overriding converter '{name}'")
            converters[name] = cls
    
    for file_path in module_specs:
        print(f"🔍 Registering file: {file_path}")
        explicit_converters = _register_converter_from_file(file_path)
        for name, cls in explicit_converters.items():
            if name in converters:
                print(f"  ⚠️  Overriding existing converter '{name}'")
            converters[name] = cls
    
    return converters


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--converter-dir', action='append', default=[],
                           help='Additional directory to scan for converters')
    pre_parser.add_argument('--converter-module', action='append', default=[],
                           help='Explicitly register a converter file (path/to/module.py or package.module)')
    
    pre_args, _ = pre_parser.parse_known_args()
    
    builtin_dir = SCRIPT_DIR / 'converters'
    
    auto_scan_dirs = []
    if 'github-opensource' in str(SCRIPT_DIR):
        parts = str(SCRIPT_DIR).split('github-opensource')
        if len(parts) == 2:
            base_path = parts[0] + 'github-opensource'
            relative_path = parts[1].lstrip('/')
            
            stub_path = Path(base_path) / 'stub_source' / relative_path / 'converters'
            if stub_path.exists() and stub_path.is_dir():
                auto_scan_dirs.append(stub_path)
    
    extra_dirs = [Path(d) for d in pre_args.converter_dir]
    module_specs = pre_args.converter_module
    
    CONVERTERS = _build_converter_registry(builtin_dir, auto_scan_dirs + extra_dirs, module_specs)
    
    if not CONVERTERS:
        print("❌ Error: No converters found. Please check your converter directories.", file=sys.stderr)
        return 1
    
    print(f"\n📋 Available formats: {', '.join(sorted(CONVERTERS.keys()))}\n")
    
    available_formats = list(CONVERTERS.keys())
    
    parser = argparse.ArgumentParser(
        description='Convert various trace formats to Optimizer standard format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[pre_parser]
    )

    parser.add_argument('-i', '--input', required=True, help='Input trace file path')
    parser.add_argument('-o', '--output', default=None, 
                        help='Output file path (default: auto-generate based on input filename and mode)')
    parser.add_argument('-f', '--format', required=True, choices=available_formats,
                        help=f'Input trace format (available: {", ".join(sorted(available_formats))})')
    parser.add_argument('--mode', default='optimizer', choices=['optimizer', 'inference'],
                        help='Output mode: optimizer (Get+Write) or inference (DialogTurn)')
    parser.add_argument('--instance-id', default='instance', help='Default instance ID')
    parser.add_argument('--instance-block-sizes', default=None,
                        help='Per-instance block sizes (format: instance1:16,instance2:32)')
    parser.add_argument('--tokenizer-path', default=None, help='Tokenizer path')
    parser.add_argument('--model-name', default=None, help='Model name')
    parser.add_argument('--model-mapping', default=None, help='Model name mapping')
    parser.add_argument('--time-field', default='time', help='Time field name for text format')
    parser.add_argument('--content-field', default='prompt_messages', help='Content field name for text format')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.output is None:
        input_dir = input_path.parent
        input_stem = input_path.stem
        suffix = '_optimizer' if args.mode == 'optimizer' else '_inference'
        output_filename = f"{input_stem}{suffix}.jsonl"
        output_path = input_dir / output_filename
        print(f"📝 Auto-generated output file: {output_path}")
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    instance_block_sizes = {}
    if args.instance_block_sizes:
        try:
            for pair in args.instance_block_sizes.split(','):
                inst_id, size = pair.split(':')
                instance_block_sizes[inst_id.strip()] = int(size.strip())
        except ValueError:
            print(f"❌ Error: Invalid --instance-block-sizes format", file=sys.stderr)
            return 1

    model_mapping = None
    if args.model_mapping:
        model_mapping = {}
        try:
            for pair in args.model_mapping.split(','):
                src_model, hf_model = pair.split(':')
                model_mapping[src_model.strip()] = hf_model.strip()
        except ValueError:
            print(f"❌ Error: Invalid --model-mapping format", file=sys.stderr)
            return 1

    try:
        converter_class = CONVERTERS.get(args.format)
        if not converter_class:
            print(f"❌ Error: Unsupported format: {args.format}", file=sys.stderr)
            return 1
        
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

        print(f"🔄 Converting {args.input} → {output_path}")
        print(f"   Format: {args.format}, Mode: {args.mode}")

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
