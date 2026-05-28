# Crash Stack Trace

KVCacheManager installs an async-signal-safe crash handler in
`CommandLine::RegisterSignal` by default. When the process is about to
be terminated by one of the following signals it dumps a readable stack
trace **in addition to** the kernel coredump:

`SIGSEGV` / `SIGABRT` / `SIGBUS` / `SIGFPE` / `SIGILL`

The handler does not replace the coredump. After printing, it reraises
the signal through the previous handler (default → kernel writes a
coredump; ASAN attached → ASAN takes over). When you have both, the
text dump is supplementary; when the coredump is lost (ulimit / bad
`core_pattern`), the text dump is the only clue you have.

## Where The Output Goes

Each crash writes two copies simultaneously:

| Location | Notes |
|---|---|
| `stderr` | Always written. Containers / systemd collect it for you |
| `logs/std_stack_trace.log` | Relative to cwd, the same `logs/` dir as the alog config; `O_APPEND`, multiple crashes accumulate |

If `logs/` cannot be created (permissions, read-only fs) the handler
gracefully degrades to stderr only — server startup is **not** blocked.

## Sample Output

```
*** KVCM CrashHandler caught signal 11 (SIGSEGV) at addr 0x..., pid 1234, tid 1234 ***
version: 0.0.1+20260518.9bfcf9d5 (commit 9bfcf9d5, built 2026-05-18 10:30:09)
si_code: 1 SEGV_MAPERR (address not mapped)
time(epoch_sec): 1779085225
Stack trace (12 frames, demangle with bin/decode_stack_trace.sh):
#0 ./bin/kv_cache_manager_bin(_ZN3foo3barEv+0x1f)[0x55d...]
#1 ./bin/kv_cache_manager_bin(_ZN3xyz4baz7processEPv+0x83)[0x55d...]
...
*** End of stack trace ***
```

Field reference:

| Field | Meaning |
|---|---|
| `signal N (NAME)` | POSIX signal number and name |
| `addr` | `siginfo_t::si_addr` — address that triggered the signal (the faulting address for SIGSEGV) |
| `pid` / `tid` | Process ID / LWP ID of the thread that crashed |
| `version` | Binary build version and git commit, used to pin code revision after the fact |
| `si_code` | Refined cause. Common values: `SEGV_MAPERR` (address not mapped), `SEGV_ACCERR` (permission), `BUS_ADRALN` (alignment), `SI_USER`/`SI_TKILL` (user-space raise/tkill), etc. |
| `time(epoch_sec)` | UNIX epoch second of the crash, useful for correlating with upstream/downstream logs |
| `#N module(symbol+0xOFF)[0xABS]` | Frame N. `module` is the binary path; `symbol+0xOFF` is the symbol and the in-symbol offset; `[0xABS]` is the runtime virtual address |

## Decoding The Trace: bin/decode_stack_trace.sh

The package ships an offline decoder. It demangles symbols with
`c++filt` and resolves `file:line` via `addr2line` + `nm` (best-effort):

```bash
# On the same host as the binary (typical dev workflow)
bin/decode_stack_trace.sh logs/std_stack_trace.log

# Cross-host: log was produced on host A, binary lives on host B
bin/decode_stack_trace.sh \
    -b /opt/kvcm/bin/kv_cache_manager_bin \
    -l /opt/kvcm/lib \
    crash.log

# Reads from stdin too
cat logs/std_stack_trace.log | bin/decode_stack_trace.sh
```

Decoded output:

```
Stack trace (12 frames, demangle with bin/decode_stack_trace.sh):
  #0   kv_cache_manager::foo::bar()  at kv_cache_manager/foo/bar.cc:42  (kv_cache_manager_bin +0x1f)
  #1   kv_cache_manager::xyz::baz::process(void*)  at kv_cache_manager/xyz/baz.cc:117  (kv_cache_manager_bin +0x83)
  ...
```

Flags:

| Flag | Purpose |
|---|---|
| `-b BINARY` | Override the main binary path embedded in the trace (cross-host analysis) |
| `-l LIB_DIR` | Directory to look up `.so` files by basename as a fallback |
| `-h` | Help |

Dependencies (script degrades gracefully when missing): `c++filt`,
`addr2line`, `nm`. All present in the default `binutils` / `gcc`
package on most Linux distros.

## CrashHandler vs Linux Coredump

| | CrashHandler output | Linux coredump |
|---|---|---|
| Producer | Current process (`backtrace_symbols_fd`) | Kernel (gated by `ulimit -c` / `core_pattern`) |
| Format | Text | ELF core file |
| Size | KB | Process memory size (GB) |
| Contents | One call stack | All threads' registers, stack, heap, mmap |
| Tooling | Text utilities + decode_stack_trace.sh | `gdb ./binary core` |

The two are complementary: CrashHandler answers "what's printable,
greppable, alertable"; the coredump answers "every variable, every
thread, fully debuggable".

## Stack Overflow Protection For Worker Threads

`sigaltstack` is a **per-thread** attribute: `Install()` only registers
the alternate signal stack on the calling thread (typically the main
thread). Threads spawned via `pthread_create` have **no** alt stack by
default. We register the crash signals with `SA_ONSTACK`, but when the
delivering thread has no alt stack the kernel falls back to that
thread's own stack — fine for the vast majority of `SIGSEGV` cases
(null-pointer dereference, UAF, …), but **broken for worker-thread
stack overflows**: the handler tries to run on an already-exhausted
stack and either truncates its output or fails to run at all.

If your worker threads have genuine stack-overflow risk (deep
recursion, large `alloca`/VLA, …), call
`CrashHandler::InstallAltStackForCurrentThread()` at the thread entry:

```cpp
#include "kv_cache_manager/common/crash_handler.h"

std::thread worker([] {
    kv_cache_manager::CrashHandler::InstallAltStackForCurrentThread();
    // ... thread body ...
});
```

Properties:
- **Idempotent**: repeated calls don't re-allocate.
- **Auto-freed**: ~64 KB per thread, registered via a `pthread_key`
  destructor so it's `free()`d automatically when the thread exits —
  no manual cleanup.
- **Fail-safe**: silent no-op on allocation failure; the worker simply
  loses overflow protection, the thread itself and other crash paths
  keep working.

Non-overflow crashes do **not** need this API — the worker's own stack
is intact, the handler runs there fine. Only opt in when you genuinely
care about stack overflow.

## Configuration

No configuration. Auto-enabled. There is no enable/disable switch by
design — this is a safety net and should not be turned off. To override
the output directory you can call `CrashHandler::Install("/your/log/dir")`
at the call site, but the default (`logs/`) already matches the alog
config so no change is normally needed.
