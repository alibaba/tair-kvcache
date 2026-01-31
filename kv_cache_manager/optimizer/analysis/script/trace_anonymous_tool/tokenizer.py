# ============================================================
# File Name: tokenizer.py
# Description:
#   - This script reads each line from file, extracts the "time" and "prompt_messages" fields,
#     tokenizes the "prompt_messages" using a specified tokenizer,
#     and writes the results (timestamp + token IDs) into a new JSONL file.
#
# Input:
#   - input file, must contain 'time' and 'prompt_messages' fields for each request.
#   - 'time' format: YYYY-MM-DD HH:MM:SS.SSSSSS
#
# Output:
#   - output file, each line contains 'timestamp' and 'token_ids'.
# ============================================================
import re
import os
import json
import argparse
import tqdm
from transformers import AutoTokenizer
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOT_YET_ANONYMIZED_DIR = os.path.join(SCRIPT_DIR, 'result/not_yet_anonymous_files')

def extract_timestamp(line: str, timestamp: str) -> float:
    obj = json.loads(line)
    ts_str = obj[timestamp]
    # return ts_str
    ts_str = ts_str.replace(',', '.')
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    return ts_str, dt.timestamp()

def extract_content(log_line, content: str):
    json_match = re.search(r'\{.*\}', log_line)
    assert json_match, "not matched successfully."
    try:
        log_data = json.loads(json_match.group())
        prompt_messages = log_data[content]
        return prompt_messages
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}") 
        return None
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return None

def is_tool_call_request(prompt_messages: dict) -> bool:
    """判断是否工具调用，根据最后一条消息是否有 tool_call_id"""
    if not isinstance(prompt_messages, dict):
        return False
    conv = prompt_messages.get("messages", [])
    if not conv:
        return False
    last_msg = conv[-1]
    return isinstance(last_msg, dict) and "tool_call_id" in last_msg

import json

def clean_messages(messages):
    """清洗 messages，处理 tool_calls 和 content"""
    if not isinstance(messages, list):
        return messages
    
    for msg in messages:
        # 处理 content：确保是字符串
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        elif not isinstance(content, str):
            msg["content"] = str(content)
        
        # 处理 tool_calls：确保 arguments 是 dict
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tool_call in msg["tool_calls"]:
                if isinstance(tool_call, dict) and isinstance(tool_call.get("function"), dict):
                    func = tool_call["function"]
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except:
                            func["arguments"] = {}
                    elif not isinstance(args, dict):
                        func["arguments"] = {}
    return messages

def clean_tools(tools):
    """清洗 tools，自动补全缺失字段而不是跳过"""
    if not isinstance(tools, list):
        print(f"Warning: tools is not list, type: {type(tools)}")
        return []
    
    cleaned = []
    for i, tool in enumerate(tools):
        # 确保 tool 是字典
        if not isinstance(tool, dict):
            print(f"Warning: tool[{i}] is not dict, converting")
            tool = {"function": {"name": f"unknown_tool_{i}", "parameters": {"type": "object", "properties": {}, "required": []}}}
        
        # 确保有 function 字段
        if 'function' not in tool:
            print(f"Warning: tool[{i}] has no 'function' key, creating default")
            tool['function'] = {"name": f"default_tool_{i}", "parameters": {"type": "object", "properties": {}, "required": []}}
        elif not isinstance(tool['function'], dict):
            print(f"Warning: tool[{i}].function is not dict, creating default")
            tool['function'] = {"name": f"default_tool_{i}", "parameters": {"type": "object", "properties": {}, "required": []}}
        
        func = tool['function']
        
        # 确保有 name
        if 'name' not in func or not isinstance(func['name'], str):
            print(f"Warning: tool[{i}].function.name is missing or invalid, setting default")
            func['name'] = f"unnamed_tool_{i}"
        
        # 确保有 description
        if 'description' not in func or not isinstance(func['description'], str):
            func['description'] = ""
        
        # 确保 parameters 结构完整
        if 'parameters' not in func:
            print(f"Warning: tool[{i}].function.parameters is missing, creating default")
            func['parameters'] = {'type': 'object', 'properties': {}, 'required': []}
        elif not isinstance(func['parameters'], dict):
            print(f"Warning: tool[{i}].function.parameters is not dict, resetting")
            func['parameters'] = {'type': 'object', 'properties': {}, 'required': []}
        
        # 补全 parameters 内部结构
        params = func['parameters']
        if 'type' not in params or not isinstance(params['type'], str):
            params['type'] = 'object'
        if 'properties' not in params or not isinstance(params['properties'], dict):
            params['properties'] = {}
        if 'required' not in params or not isinstance(params['required'], list):
            params['required'] = []
        
        # 确保 tool 有 type 字段
        if 'type' not in tool or tool['type'] != 'function':
            tool['type'] = 'function'
            
        cleaned.append(tool)
    return cleaned


def get_tokenizer_path(model_name, tokenizer_path):
    model_dir = f"{SCRIPT_DIR}/model/{model_name}"
    chosen_tokenizer_path = tokenizer_path
    if model_name is not None and os.path.exists(model_dir):
        chosen_tokenizer_path = model_dir
    print(f"chosen tokenizer path: {chosen_tokenizer_path}")
    return chosen_tokenizer_path

def gen_token_ids_file(
    file_path, tokenizer: AutoTokenizer,
    timestamp_field, content_field,
    content_extractor=extract_content, 
    timestamp_extractor=extract_timestamp
):
    trace_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(f"{NOT_YET_ANONYMIZED_DIR}", exist_ok=True)
    output_file = f"{NOT_YET_ANONYMIZED_DIR}/tokenids_{trace_name}.jsonl"

    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f); f.seek(0)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line in tqdm.tqdm(f, total=total_lines, desc=f"Tokenizing {trace_name}"):
                ts_str, timestamp = timestamp_extractor(line, timestamp_field)
                prompt_messages = content_extractor(line, content_field)
                if "tools" in prompt_messages and prompt_messages["tools"]:
                    prompt_messages["messages"] = clean_messages(prompt_messages["messages"])
                    prompt_messages["tools"] = clean_tools(prompt_messages["tools"])
                    token_ids = tokenizer.apply_chat_template(
                        conversation=prompt_messages["messages"],
                        tools=prompt_messages["tools"],
                        tokenize=True
                    )
                else:
                    token_ids = tokenizer.encode(f"{prompt_messages}")
                
                is_tool_call = is_tool_call_request(prompt_messages)
                item = {
                    "ts_str": ts_str,
                    "timestamp": timestamp,
                    "token_ids": token_ids,
                    "is_tool_call": is_tool_call
                }
                out_f.write(json.dumps(item) + '\n')
    print(f"Token ids written to {output_file}")
    return output_file

if __name__ == '__main__':
    """ python tokenizer.py --file_path <input_file> """
    parser = argparse.ArgumentParser(description='Extract token ids from log')
    parser.add_argument('--file_path', type=str, required=True)
    # field name
    parser.add_argument('--time_field', type=str, default='time')
    parser.add_argument('--content_field', type=str, default='prompt_messages')
    # tokenizer path
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--tokenizer_path', type=str, default='model/deepseek-v3')
    args = parser.parse_args()

    gen_token_ids_file(
        file_path=args.file_path,
        tokenizer=AutoTokenizer.from_pretrained(get_tokenizer_path(args.model_name, args.tokenizer_path)),
        timestamp_field=args.time_field,
        content_field=args.content_field,
    )
