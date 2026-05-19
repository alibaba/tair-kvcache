#pragma once

#include <string>

namespace kv_cache_manager {

class CrashHandler {
public:
    static void Install(const std::string &log_dir = "");

    static bool IsInstalled();

    // 给当前线程装一个 alternate signal stack，用于 SIGSEGV 等 crash signal
    // 在线程**栈溢出**时的兜底输出。Install() 只装了主线程，pthread_create
    // 出来的 worker 线程默认没有 alt stack —— 这些线程发生栈溢出时，handler
    // 在已经爆掉的 worker stack 上跑不起来，堆栈输出可能丢失或截断。
    //
    // 调用约定：在 worker 线程入口（pthread 回调的开头）调一次即可，幂等。
    // 内存按线程分配（约 64 KB），随线程退出自动释放。栈溢出之外的崩溃
    // （nullptr 解引用、UAF 等）不需要这个 —— worker stack 本身没坏，
    // handler 正常跑。仅在线程有强烈的栈溢出担忧时调用。
    static void InstallAltStackForCurrentThread();
};

} // namespace kv_cache_manager
