"""
Tokenizer加载工具

统一的tokenizer加载逻辑:
1. 如果指定model_name且./model/{model_name}存在,使用它
2. 否则使用tokenizer_path
3. 直接调用AutoTokenizer.from_pretrained()
"""

import os
from pathlib import Path
from typing import Optional, Any

# ========================================
# 延迟加载标志
# ========================================
_TRANSFORMERS_INITIALIZED = False
_HAS_TRANSFORMERS = False
_AutoTokenizer = None


def _ensure_transformers_loaded():
    """延迟加载 transformers（只在真正需要时加载）"""
    global _TRANSFORMERS_INITIALIZED, _HAS_TRANSFORMERS, _AutoTokenizer
    
    if _TRANSFORMERS_INITIALIZED:
        return _HAS_TRANSFORMERS
    
    _TRANSFORMERS_INITIALIZED = True
    
    # 设置HuggingFace镜像 (必须在import transformers之前)
    if 'HF_ENDPOINT' not in os.environ:
        # 默认使用国内镜像
        default_mirror = 'https://hf-mirror.com'
        os.environ['HF_ENDPOINT'] = default_mirror
        print(f"🌐 Using default HuggingFace mirror: {default_mirror}")
        print(f"   (Set HF_ENDPOINT='' before running to use official source)")
    
    try:
        from transformers import AutoTokenizer as _AT
        _AutoTokenizer = _AT
        _HAS_TRANSFORMERS = True
    except ImportError:
        _HAS_TRANSFORMERS = False
    
    return _HAS_TRANSFORMERS


def get_tokenizer(
    tokenizer_path: str,
    model_name: Optional[str] = None
) -> Any:
    """
    加载tokenizer 

    逻辑:
    1. 如果指定model_name且./model/{model_name}存在 → 使用它
    2. 否则使用tokenizer_path
    3. 直接传给AutoTokenizer.from_pretrained(),让它自动判断:
       - 本地路径 → 直接加载
       - HuggingFace模型ID → 自动下载 (通过HF_ENDPOINT环境变量控制镜像)

    环境变量:
        HF_ENDPOINT: HuggingFace镜像地址 (默认: https://hf-mirror.com)
                    设置为空可使用官方源

    Args:
        tokenizer_path: Tokenizer路径 (本地路径或HuggingFace模型ID)
        model_name: 模型名称 (可选,用于在./model/目录中查找本地缓存)

    Returns:
        AutoTokenizer实例

    Examples:
        # 本地路径
        tokenizer = get_tokenizer("model/deepseek-v3")

        # HuggingFace模型ID (使用默认镜像)
        tokenizer = get_tokenizer("Qwen/Qwen2.5-7B-Instruct")

        # 使用官方源 (设置环境变量为空)
        export HF_ENDPOINT=""
        tokenizer = get_tokenizer("Qwen/Qwen2.5-7B-Instruct")

        # 使用自定义镜像
        export HF_ENDPOINT=https://hf-mirror.com
        tokenizer = get_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    """
    # 延迟加载 transformers
    if not _ensure_transformers_loaded():
        raise ImportError(
            "transformers library is required. "
            "Install it with: pip install transformers"
        )

    # 确定最终路径
    script_dir = Path(__file__).parent.parent  # trace_converter目录

    # 优先级1: 如果指定model_name且./model/{model_name}存在,用它
    if model_name is not None:
        model_dir = script_dir / "model" / model_name
        if model_dir.exists():
            chosen_path = str(model_dir)
            print(f"✅ Using local tokenizer: {chosen_path}")
            try:
                tokenizer = _AutoTokenizer.from_pretrained(chosen_path)
                return tokenizer
            except Exception as e:
                print(f"❌ Error loading tokenizer from '{chosen_path}': {e}")
                raise

    # 优先级2: 使用tokenizer_path
    if not tokenizer_path:
        raise ValueError("tokenizer_path is required (or ensure ./model/{model_name} exists)")

    chosen_path = tokenizer_path

    # 判断是本地路径还是HF模型ID
    is_local = os.path.exists(chosen_path) or os.path.isabs(chosen_path)
    if is_local:
        print(f"✅ Loading tokenizer from local path: {chosen_path}")
    else:
        mirror = os.environ.get('HF_ENDPOINT', '')
        print(f"📥 Downloading tokenizer from HuggingFace: {chosen_path}")
        if mirror:
            print(f"   (via mirror: {mirror})")

    # 直接传给AutoTokenizer.from_pretrained()
    # 它会自动判断是本地路径还是HF模型ID
    try:
        tokenizer = _AutoTokenizer.from_pretrained(chosen_path)
        print(f"✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer from '{chosen_path}': {e}")
        if not is_local and not os.environ.get('HF_ENDPOINT'):
            print(f"💡 Tip: Set HF_ENDPOINT environment variable to use a mirror")
        raise


def smart_tokenize(
    tokenizer: Any,
    content: Any,
    use_chat_template: bool = True
) -> list:
    """
    智能tokenization,自动选择最佳方法

    策略:
    1. 如果content是dict且包含messages: 使用apply_chat_template
    2. 如果content是list: 使用apply_chat_template
    3. 其他情况: 使用encode

    Args:
        tokenizer: AutoTokenizer实例
        content: 待tokenize的内容
        use_chat_template: 是否尝试使用chat template

    Returns:
        Token IDs列表

    Examples:
        # 对话格式
        content = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        }
        token_ids = smart_tokenize(tokenizer, content)

        # 纯文本
        token_ids = smart_tokenize(tokenizer, "Hello world")
    """
    # ========================================
    # 策略1: 对话格式 (dict with messages)
    # ========================================
    if use_chat_template and isinstance(content, dict):
        messages = content.get("messages")
        tools = content.get("tools")

        if messages and isinstance(messages, list):
            try:
                # 清洗messages (确保content是字符串)
                cleaned_messages = _clean_messages(messages)

                # 如果有tools,一起传入
                if tools and isinstance(tools, list):
                    cleaned_tools = _clean_tools(tools)
                    result = tokenizer.apply_chat_template(
                        conversation=cleaned_messages,
                        tools=cleaned_tools,
                        tokenize=True,
                        add_generation_prompt=False
                    )
                else:
                    result = tokenizer.apply_chat_template(
                        conversation=cleaned_messages,
                        tokenize=True,
                        add_generation_prompt=False
                    )

                # 处理 BatchEncoding 对象
                if hasattr(result, 'input_ids'):
                    return result.input_ids
                return result if isinstance(result, list) else list(result)
            except Exception as e:
                print(f"Warning: apply_chat_template failed: {e}, falling back to encode")
                # Fallback to encode
                pass

    # ========================================
    # 策略2: 消息列表格式
    # ========================================
    if use_chat_template and isinstance(content, list):
        try:
            cleaned_messages = _clean_messages(content)
            result = tokenizer.apply_chat_template(
                conversation=cleaned_messages,
                tokenize=True,
                add_generation_prompt=False
            )
            # 处理 BatchEncoding 对象
            if hasattr(result, 'input_ids'):
                return result.input_ids
            return result if isinstance(result, list) else list(result)
        except Exception as e:
            print(f"Warning: apply_chat_template failed: {e}, falling back to encode")
            # Fallback to encode
            pass

    # ========================================
    # 策略3: 纯文本格式 (fallback)
    # ========================================
    text = str(content)
    result = tokenizer.encode(text, add_special_tokens=True)

    # 处理 BatchEncoding 对象
    if hasattr(result, 'input_ids'):
        return result.input_ids
    return result if isinstance(result, list) else list(result)


def _clean_messages(messages: list) -> list:
    """
    清洗messages,确保格式正确

    处理:
    1. content字段: 确保是字符串
    2. tool_calls.function.arguments: 确保是dict
    """
    if not isinstance(messages, list):
        return messages

    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        # 复制消息避免修改原始数据
        cleaned_msg = msg.copy()

        # 处理content: 确保是字符串
        content = cleaned_msg.get("content")
        if content is None:
            cleaned_msg["content"] = ""
        elif not isinstance(content, str):
            cleaned_msg["content"] = str(content)

        # 处理tool_calls: 确保arguments是dict
        if cleaned_msg.get("role") == "assistant" and isinstance(cleaned_msg.get("tool_calls"), list):
            tool_calls = []
            for tool_call in cleaned_msg["tool_calls"]:
                if isinstance(tool_call, dict) and isinstance(tool_call.get("function"), dict):
                    func = tool_call["function"].copy()
                    args = func.get("arguments")

                    # 如果arguments是字符串,尝试解析为dict
                    if isinstance(args, str):
                        try:
                            import json
                            func["arguments"] = json.loads(args)
                        except BaseException:
                            func["arguments"] = {}
                    elif not isinstance(args, dict):
                        func["arguments"] = {}

                    tool_call = tool_call.copy()
                    tool_call["function"] = func
                    tool_calls.append(tool_call)

            cleaned_msg["tool_calls"] = tool_calls

        cleaned.append(cleaned_msg)

    return cleaned


def _clean_tools(tools: list) -> list:
    """
    清洗tools,自动补全缺失字段

    确保每个tool有:
    - type: "function"
    - function.name: str
    - function.description: str
    - function.parameters: dict with type/properties/required
    """
    if not isinstance(tools, list):
        return []

    cleaned = []
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue

        # 复制避免修改原始数据
        cleaned_tool = tool.copy()

        # 确保有type字段
        if cleaned_tool.get("type") != "function":
            cleaned_tool["type"] = "function"

        # 确保有function字段
        if "function" not in cleaned_tool or not isinstance(cleaned_tool["function"], dict):
            cleaned_tool["function"] = {}

        func = cleaned_tool["function"].copy()

        # 确保有name
        if "name" not in func or not isinstance(func["name"], str):
            func["name"] = f"tool_{i}"

        # 确保有description
        if "description" not in func or not isinstance(func["description"], str):
            func["description"] = ""

        # 确保parameters结构完整
        if "parameters" not in func or not isinstance(func["parameters"], dict):
            func["parameters"] = {}

        params = func["parameters"].copy()
        if "type" not in params:
            params["type"] = "object"
        if "properties" not in params or not isinstance(params["properties"], dict):
            params["properties"] = {}
        if "required" not in params or not isinstance(params["required"], list):
            params["required"] = []

        func["parameters"] = params
        cleaned_tool["function"] = func
        cleaned.append(cleaned_tool)

    return cleaned
