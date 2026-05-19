#!/bin/bash
# decode_stack_trace.sh — 把 KVCM CrashHandler 输出的 std_stack_trace.log 中的
# C++ mangled 符号 demangle，并尽量用 addr2line 解出源码 file:line。
#
# 输入格式（每帧一行）：
#   <path>(SYMBOL+0xOFFSET)[0xABS]    带符号 + 符号内偏移 + 运行时绝对地址
#   <path>(+0xOFFSET)[0xABS]          仅文件偏移 + 绝对地址（一般是动态库内私有符号）
#   <path>[0xABS]                     仅绝对地址（main binary 中的 static 符号）
#
# 解析策略：
#   1) demangle：所有 _Z 开头的 mangled 符号统统过 c++filt，能解就解。
#   2) addr2line（best-effort）：
#      - 优先用 (SYMBOL+0xOFFSET)：把 SYMBOL 在二进制里的地址（nm 查）加上 OFFSET，
#        得到 binary 内偏移，喂给 addr2line。这对 PIE 二进制也成立。
#      - 退化路径：直接拿 [0xABS] 喂 addr2line。对非 PIE 二进制可用，对 PIE 通常会
#        得到 ??:?，那就跳过。
#   3) 二进制路径优先用栈帧里写的那个；不可读时回退到 -b/--binary 或 -l/--lib-dir。
#
# 这是离线工具，不要求严格性能，逐行处理。

set -uo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [-b BINARY] [-l LIB_DIR] [-h] [LOG_FILE]

Decode a KVCM std_stack_trace.log:
  - demangle C++ symbols (c++filt)
  - resolve source file:line (addr2line, best-effort)

Options:
  -b BINARY     Override main binary path (used when frame's embedded path
                doesn't exist on this host, e.g. logs collected from another box)
  -l LIB_DIR    Directory to look up shared libraries by basename when the
                embedded path can't be opened (fallback before -b)
  -h            Show this help

Input:
  LOG_FILE      std_stack_trace.log path; defaults to stdin

Examples:
  $(basename "$0") logs/std_stack_trace.log
  $(basename "$0") -b /opt/kvcm/bin/kv_cache_manager_bin std_stack_trace.log
  cat std_stack_trace.log | $(basename "$0")
EOF
}

BINARY_OVERRIDE=""
LIB_DIR=""
while getopts ":b:l:h" opt; do
    case "$opt" in
        b) BINARY_OVERRIDE="$OPTARG" ;;
        l) LIB_DIR="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
        :)  echo "option -$OPTARG requires an argument" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
LOG_INPUT="${1:-/dev/stdin}"

have() { command -v "$1" >/dev/null 2>&1; }

if ! have c++filt; then
    echo "[warn] c++filt not in PATH; symbols will stay mangled" >&2
fi
if ! have addr2line; then
    echo "[warn] addr2line not in PATH; file:line resolution disabled" >&2
fi
if ! have nm; then
    echo "[warn] nm not in PATH; SYMBOL+OFFSET resolution disabled" >&2
fi

# 给定栈帧路径，返回本机能打开的文件路径（找不到返回空）。
resolve_binary() {
    local p="$1"
    if [[ -r "$p" ]]; then
        printf '%s' "$p"
        return 0
    fi
    if [[ -n "$LIB_DIR" ]]; then
        local cand="$LIB_DIR/$(basename "$p")"
        if [[ -r "$cand" ]]; then
            printf '%s' "$cand"
            return 0
        fi
    fi
    if [[ -n "$BINARY_OVERRIDE" && "$(basename "$p")" == "$(basename "$BINARY_OVERRIDE")" ]]; then
        printf '%s' "$BINARY_OVERRIDE"
        return 0
    fi
    return 1
}

# 把 c++filt 单次输出 trim 一下；mangled 输入失败时 c++filt 会原样返回。
demangle() {
    local s="$1"
    if [[ -z "$s" ]] || ! have c++filt; then
        printf '%s' "$s"
        return
    fi
    c++filt -- "$s" 2>/dev/null
}

# 在 binary 中查 symbol 的地址（十六进制，无 0x 前缀）。找不到返回非 0。
# 假定 GNU nm 输出格式（"ADDR TYPE SYMBOL"）。BSD nm 列序不同，本脚本
# 只兼容 Linux/binutils 默认的 GNU nm；macOS/FreeBSD 需自行替换为
# `nm -P`（POSIX 格式）。
nm_symbol_addr() {
    local bin="$1" sym="$2"
    have nm || return 1
    nm --defined-only "$bin" 2>/dev/null \
        | awk -v s="$sym" '$3 == s { print $1; exit }'
}

# addr2line -fCp 输出形如 "function_name at file:line"。这里只取第一行，
# 不展开 inline 链。完全没解到（函数和位置都未知）才返回非 0；只有函数没有
# 位置也算成功（很多动态库函数 stripped 了 debug info 但保留 dynsym）。
addr2line_at() {
    local bin="$1" hex="$2"
    have addr2line || return 1
    local out
    out=$(addr2line -e "$bin" -fCp "$hex" 2>/dev/null | head -n 1) || return 1
    case "$out" in
        ""|"?? ??:?"|"?? ??:0") return 1 ;;
    esac
    printf '%s' "$out"
}

# 把 addr2line 一行结果拆成 "func" 和 "file:line"。任意一段未解到就置空。
# 也把 bazel 编译时常见的 /proc/self/cwd/ 前缀去掉，让源码路径相对仓库根。
split_addr2line() {
    local out="$1"
    local func="" loc=""
    if [[ "$out" == *" at "* ]]; then
        func="${out% at *}"
        loc="${out##* at }"
    else
        func="$out"
    fi
    [[ "$func" == "??" ]] && func=""
    # 各种"位置无效"的写法都吞掉：
    #   ""           完全无输出
    #   ":?" / ":0"  文件名空
    #   "??:?" 等    addr2line 标准的 unknown 标记
    case "$loc" in
        ""|":"*|*"??:"*) loc="" ;;
    esac
    loc="${loc#/proc/self/cwd/}"
    printf '%s\t%s' "$func" "$loc"
}

# 解一帧并打印；不是栈帧的行原样输出。
decode_frame() {
    local line="$1"
    local path="" sym="" off="" abs=""
    local frame_no=""

    # 新格式（CrashHandler 直接生成）每帧前面带 "#N "，剥掉再走原解析。
    # 没有前缀也兼容（旧日志或测试环境）：用脚本本地维护的 FRAME_IDX。
    local re_idx='^#([0-9]+)[[:space:]]+(.*)$'
    if [[ "$line" =~ $re_idx ]]; then
        frame_no="${BASH_REMATCH[1]}"
        line="${BASH_REMATCH[2]}"
    fi

    # 把正则放进变量避免 bash 在 [[ =~ ... ]] 内的词法分析问题
    local re_full='^([^(]+)\(([^+()]+)\+0x([0-9a-fA-F]+)\)\[0x([0-9a-fA-F]+)\]$'
    local re_off='^([^(]+)\(\+0x([0-9a-fA-F]+)\)\[0x([0-9a-fA-F]+)\]$'
    local re_abs='^([^[]+)\[0x([0-9a-fA-F]+)\]$'

    if [[ "$line" =~ $re_full ]]; then
        path="${BASH_REMATCH[1]}"
        sym="${BASH_REMATCH[2]}"
        off="${BASH_REMATCH[3]}"
        abs="${BASH_REMATCH[4]}"
    elif [[ "$line" =~ $re_off ]]; then
        path="${BASH_REMATCH[1]}"
        off="${BASH_REMATCH[2]}"
        abs="${BASH_REMATCH[3]}"
    elif [[ "$line" =~ $re_abs ]]; then
        path="${BASH_REMATCH[1]}"
        abs="${BASH_REMATCH[2]}"
    else
        printf '%s\n' "$line"
        return
    fi

    local pretty_sym=""
    if [[ -n "$sym" ]]; then
        pretty_sym=$(demangle "$sym")
    fi

    local resolved_bin=""
    resolved_bin=$(resolve_binary "$path" || true)

    # 三路尝试 addr2line：
    #  1. SYMBOL+OFFSET → nm 查 SYMBOL 在 binary 的地址 + OFFSET（PIE 友好）
    #  2. 直接喂 [0xABS]（对 non-PIE 二进制有效）
    #  3. 直接喂 0xOFFSET（一般无效，但便宜，兜底）
    local a2l_out=""
    if [[ -n "$resolved_bin" ]]; then
        if [[ -n "$sym" && -n "$off" ]]; then
            local sym_addr
            sym_addr=$(nm_symbol_addr "$resolved_bin" "$sym" || true)
            if [[ -n "$sym_addr" ]]; then
                local target_hex
                target_hex=$(printf '0x%x' $((0x$sym_addr + 0x$off)))
                a2l_out=$(addr2line_at "$resolved_bin" "$target_hex" || true)
            fi
        fi
        if [[ -z "$a2l_out" && -n "$abs" ]]; then
            a2l_out=$(addr2line_at "$resolved_bin" "0x$abs" || true)
        fi
        if [[ -z "$a2l_out" && -n "$off" ]]; then
            a2l_out=$(addr2line_at "$resolved_bin" "0x$off" || true)
        fi
    fi

    local a2l_func="" a2l_source=""
    if [[ -n "$a2l_out" ]]; then
        IFS=$'\t' read -r a2l_func a2l_source < <(split_addr2line "$a2l_out")
    fi

    # 名字优先级：原 SYMBOL（demangled） > addr2line 函数名 > "??"
    local display_name="${pretty_sym:-${a2l_func:-??}}"

    # 地址段：用相对偏移更短；没有就用绝对地址
    local addr_part=""
    if [[ -n "$off" ]]; then
        addr_part="+0x$off"
    elif [[ -n "$abs" ]]; then
        addr_part="[0x$abs]"
    fi

    local bin_basename
    bin_basename=$(basename "$path")

    # 序号优先用从输入剥下来的；没有就用脚本本地计数（兼容旧日志）
    local out_idx="${frame_no:-$FRAME_IDX}"

    if [[ -n "$a2l_source" ]]; then
        printf '  #%-3d %s  at %s  (%s %s)\n' "$out_idx" "$display_name" "$a2l_source" "$bin_basename" "$addr_part"
    else
        printf '  #%-3d %s  (%s %s)\n' "$out_idx" "$display_name" "$bin_basename" "$addr_part"
    fi
    FRAME_IDX=$((FRAME_IDX + 1))
}

FRAME_IDX=0

while IFS= read -r line; do
    case "$line" in
        "Stack trace ("*)
            # 一个新的崩溃栈开始，编号从 0 重新开始
            FRAME_IDX=0
            printf '%s\n' "$line"
            ;;
        "*** "*|"version:"*|"si_code:"*|"time("*|"")
            printf '%s\n' "$line"
            ;;
        *)
            decode_frame "$line"
            ;;
    esac
done < "$LOG_INPUT"
