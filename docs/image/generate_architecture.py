#!/usr/bin/env python3
"""Render the bilingual, 16:9 README / presentation architecture diagrams.

Run from any directory:
    python3 docs/image/generate_architecture.py

SVG generation uses only the Python standard library. When rsvg-convert is
available, also export 3840 x 2160 PNGs. Install Alibaba PuHuiTi 2.0 or Noto Sans
CJK SC for consistent Chinese typography. SVG text remains editable.

This is a runtime / responsibility overview, not an exhaustive build dependency
graph. See docs/design/module_architecture.md for the engineering-level graph.
"""

import argparse
from html import escape
from pathlib import Path
import shutil
import subprocess


INK = "#152B48"
MUTED = "#566B83"
BLUE = "#2865D6"
TEAL = "#058475"
ORANGE = "#C96A20"
PURPLE = "#8060B4"
BORDER = "#DCE4EE"
FONT = "'Alibaba PuHuiTi 2.0', 'Noto Sans CJK SC', 'Microsoft YaHei', Arial, sans-serif"

COPY = {
    "zh": {
        "subtitle": "中心化元数据管理 · 去中心化数据搬运",
        "engines": "推理服务",
        "engine_report": "推理侧 L1.5 / L2 缓存状态上报",
        "client_desc": "Python 引擎接入 · C++ SDK / pybind",
        "metadata_desc": "查询 / 申请位置 · 完成写入确认",
        "metadata_only": "仅元数据面 · Java / 轻量 Python",
        "transfer_desc": "引擎显存 / 内存 ↔ 存储后端",
        "server_desc": "统一管理 KVCache 的位置、状态与容量",
        "service": "gRPC / HTTP · Meta / Admin / Debug · Leader 门禁",
        "manager_1": "匹配查询 · 两阶段写入 · 后端选择",
        "manager_2": "配额回收 · 后台 GC · 分层迁移",
        "manager_3": "推理侧缓存状态管理 · ReportEvent / GetHostCacheState",
        "meta": ["Block → CacheLocation", "索引 · 查询缓存", "local / Redis"],
        "config": ["实例 · 配额 · 存储", "注册表 · HA 选主", "LeaderElector"],
        "storage": ["存储后端统一抽象", "健康 · 水位 · 容量", "EventReportBackend"],
        "support": ["事件总线", "可观测性", "proto / gRPC", "日志 · 上下文 · 发现"],
        "ops": "运维与管控",
        "ops_lines": ["kvcm_ops · Admin CLI", "存储 / 实例 / Group 管理", "配额 · 配置 · 集群状态"],
        "optimizer": "缓存仿真与优化",
        "optimizer_lines": ["Trace 回放 · 策略与容量评估", "LiteHit · 多容量命中率", "在线 TraceQuery"],
        "hisim": "LLM 推理仿真",
        "standalone": "独立运行",
        "hisim_lines": ["CPU 回放 · 无需 Manager 运行时", "预测 TTFT / TPOT / 吞吐"],
        "storage_title": "可插拔 KVCache 存储",
        "report_title": "ReportEvent · 缓存状态上报",
        "report_desc": "L1.5 / L2 · 增量 / 快照 / 心跳",
        "metadata": "元数据面",
        "data": "数据面",
        "control": "管控面",
        "events": "事件流",
        "backend": "后端管理",
        "direct": "KVCache 直连读写 · 不经过 KVCM",
        "isolation": "复用边界：相同 instance_id 内，跨请求 / 跨节点",
        "optional": "* Mooncake / VCNS HF3FS 为可选构建后端",
    },
    "en": {
        "subtitle": "Centralized metadata management · Decentralized data transfer",
        "engines": "Inference services",
        "engine_report": "L1.5 / L2 cache state reporting",
        "client_desc": "Python integration · C++ / pybind",
        "metadata_desc": "Match / reserve · confirm writes",
        "metadata_only": "Metadata only · Java / Python",
        "transfer_desc": "GPU / host memory ↔ storage",
        "server_desc": "Global KVCache locations, state and capacity",
        "service": "gRPC / HTTP · Meta / Admin / Debug · Leader gate",
        "manager_1": "Location matching · two-phase writes · backend selection",
        "manager_2": "Quota & reclaim · background GC · tiered migration",
        "manager_3": "Inference-side cache state · ReportEvent / GetHostCacheState",
        "meta": ["Block → CacheLocation", "Index · query cache", "local / Redis"],
        "config": ["Instances · quotas", "Registry · HA election", "LeaderElector"],
        "storage": ["Backend abstraction", "Health · watermarks", "EventReportBackend"],
        "support": ["Event bus", "Observability", "proto / gRPC", "Logs · Redis · discovery"],
        "ops": "Operations",
        "ops_lines": ["kvcm_ops · Admin CLI", "Storage / Instance / Group", "Quota · config · cluster state"],
        "optimizer": "Cache simulation & optimization",
        "optimizer_lines": ["Trace replay · policy & capacity", "LiteHit · multi-capacity hit rate", "Online TraceQuery"],
        "hisim": "LLM inference simulation",
        "standalone": "Standalone",
        "hisim_lines": ["CPU replay · no Manager runtime", "Predict TTFT / TPOT / throughput"],
        "storage_title": "Pluggable KVCache storage",
        "report_title": "ReportEvent · cache state",
        "report_desc": "L1.5/L2 · delta · snapshot · heartbeat",
        "metadata": "Metadata",
        "data": "Data",
        "control": "Control",
        "events": "Events",
        "backend": "Backend ops",
        "direct": "Direct KVCache I/O · bypasses KVCM",
        "isolation": "Reuse scope: same instance_id, across requests / nodes",
        "optional": "* Mooncake / VCNS HF3FS are optional build-time backends",
    },
}


class Diagram:
    def __init__(self, language, title="Tair KVCache Manager — system architecture", description=None):
        self.language = language
        self.c = COPY[language]
        if description is None:
            description = (
                "Inference clients send metadata requests to the HA-capable KVCM server. "
                "TransferClient moves KVCache directly between engine memory and supported "
                "storage systems. Inference services report their L1.5/L2 cache state through "
                "Client/Connector ReportEvent calls on the metadata plane. Service and CacheManager "
                "update indexes and EventReportBackend state. ReportEvent carries metadata only. Admin tools "
                "manage configuration; Optimizer subscribes to cache events. HiSim runs "
                "independently. Cache reuse is isolated by instance_id."
            )
        self.parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" '
            'viewBox="0 0 1920 1080" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<defs>',
        ]
        for name, color in [("blue", BLUE), ("teal", TEAL), ("orange", ORANGE),
                            ("purple", PURPLE), ("gray", "#97A9BE")]:
            self.parts.append(
                f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 1 1 L 9 5 L 1 9 Z" fill="{color}"/></marker>'
            )
        self.parts.extend([
            '<linearGradient id="manager" x1="0" y1="0" x2="1" y2="1">'
            '<stop stop-color="#EDF4FF"/><stop offset="1" stop-color="#F7FAFF"/>'
            '</linearGradient>',
            '</defs>',
            f'<g font-family="{FONT}">',
        ])

    def rect(self, x, y, w, h, fill="white", stroke=BORDER, radius=16, sw=1.5):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, value, size=22, color=INK, weight=400, anchor="start", max_width=None):
        width_attr = f' data-max-width="{max_width}"' if max_width else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}"{width_attr}>{escape(value)}</text>'
        )

    def line(self, points, color="#97A9BE", width=2, marker=None, dashed=False, both=False):
        attrs = f' marker-end="url(#{marker})"' if marker else ""
        if marker and both:
            attrs += f' marker-start="url(#{marker})"'
        if dashed:
            attrs += ' stroke-dasharray="7 6"'
        self.parts.append(
            f'<path d="{points}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{attrs}/>'
        )

    def pill(self, x, y, w, label, fill, color, size=18):
        self.rect(x, y, w, 32, fill, "none", 16)
        self.text(x + w / 2, y + 23, label, size, color, 500, "middle", w - 16)

    def render(self):
        c = self.c
        self.rect(0, 0, 1920, 1080, "#F7F9FC", "none", 0)
        self.rect(56, 57, 7, 78, "#F18536", "none", 3)
        self.text(83, 93, "Tair KVCache Manager", 44, weight=700)
        self.text(84, 138, c["subtitle"], 25, MUTED, max_width=1500)
        self.text(1864, 88, "SYSTEM ARCHITECTURE", 18, MUTED, 500, "end")
        self.text(1864, 120, "KVCM  /  MODULE OVERVIEW", 16, "#8595A9", anchor="end")

        # Three reading columns: inference access, metadata service, operations / analysis.
        self.rect(56, 202, 384, 128)
        self.text(80, 235, c["engines"], 28, weight=600)
        self.text(80, 267, "vLLM · SGLang · TensorRT-LLM", 22, MUTED, max_width=336)
        self.text(80, 293, "RTP-LLM", 22, MUTED)
        self.text(80, 318, c["engine_report"], 16, BLUE, max_width=336)
        self.line("M 248 330 V 363", marker="gray")

        self.rect(56, 364, 384, 498)
        self.text(80, 407, "Client / Connector", 29, weight=600)
        self.text(80, 445, "py_connector + client", 23, BLUE, 500)
        self.text(80, 475, c["client_desc"], 19, MUTED, max_width=336)

        self.rect(76, 488, 344, 96, "#F0F5FF", "#D2E0FA", 12)
        self.text(94, 518, "MetaClient · gRPC", 23, BLUE, 600)
        self.text(94, 548, "+ Python HTTP", 22, BLUE)
        self.text(94, 575, c["metadata_desc"], 18, MUTED, max_width=308)

        # Cache state originates in inference services, alongside their other metadata calls.
        self.rect(76, 600, 344, 78, "#F0F5FF", "#B7CDF4", 12)
        self.text(94, 630, c["report_title"], 22, BLUE, 600, max_width=308)
        self.text(94, 660, c["report_desc"], 17, MUTED, max_width=308)

        self.rect(76, 694, 344, 66, "#F7F9FD", BORDER, 12)
        self.text(94, 720, "client_java / manager_client", 22, weight=500, max_width=308)
        self.text(94, 748, c["metadata_only"], 18, MUTED, max_width=308)

        self.rect(76, 776, 344, 68, "#EAF8F3", "#BBE2D7", 12)
        self.text(94, 803, "TransferClient", 25, TEAL, 600)
        self.text(94, 832, c["transfer_desc"], 18, MUTED, max_width=308)

        # The HA boundary contains five core modules plus shared infrastructure.
        self.rect(560, 202, 824, 660, "#FFFFFF", "#B9CFEA", 20, 2)
        self.text(584, 243, "KVCM Server", 31, weight=600)
        self.pill(1138, 218, 222, "Leader + Followers", "#EAF0FB", BLUE, 19)
        self.text(584, 274, c["server_desc"], 21, MUTED, max_width=764)

        self.rect(584, 294, 776, 92, "#F5F8FD", "#D4E0F0", 12)
        self.text(606, 328, "service", 27, weight=600)
        self.text(606, 363, c["service"], 21, MUTED, max_width=732)
        self.line("M 972 386 V 410", marker="gray")

        self.rect(584, 410, 776, 160, "url(#manager)", "#B7CDF4", 12)
        self.rect(584, 426, 5, 128, BLUE, "none", 2)
        self.text(606, 446, "manager", 29, BLUE, 600)
        self.text(1336, 445, "CacheManager", 21, BLUE, 500, "end")
        self.text(606, 482, c["manager_1"], 23, max_width=732)
        self.text(606, 515, c["manager_2"], 23, max_width=732)
        self.text(606, 548, c["manager_3"], 20, MUTED, max_width=732)

        self.line("M 972 570 V 586 H 706 V 602", marker="gray")
        self.line("M 972 586 V 602", marker="gray")
        self.line("M 972 586 H 1238 V 602", marker="gray")
        for x, name, key in [(584, "meta", "meta"), (850, "config", "config"),
                             (1116, "data_storage", "storage")]:
            self.rect(x, 602, 244, 150, "#FAFBFE", "#D8E2EF", 12)
            self.text(x + 18, 637, name, 26 if key != "storage" else 24, weight=600)
            for i, label in enumerate(c[key]):
                self.text(x + 18, 675 + i * 29, label, 19, MUTED, max_width=212)

        self.rect(584, 778, 776, 66, "#F0F4F9", "none", 10)
        for x, name, desc in zip([604, 794, 984, 1174],
                                  ["event", "metrics", "protocol", "common"], c["support"]):
            self.text(x, 804, name, 22, weight=600)
            self.text(x, 830, desc, 16, MUTED, max_width=176)

        # Operations are separate from inference metadata requests.
        self.rect(1472, 202, 392, 184, "#FFFBF5", "#EAD8C4")
        self.text(1496, 241, c["ops"], 28, ORANGE, 600)
        for i, label in enumerate(c["ops_lines"]):
            self.text(1496, 280 + i * 33, label, 22 if i == 0 else 21,
                      INK if i == 0 else MUTED, max_width=344)

        self.rect(1472, 432, 392, 230, "#FCFAFF", "#DFD5EF")
        self.text(1496, 472, "optimizer", 29, PURPLE, 600)
        self.text(1496, 508, c["optimizer"], 21, MUTED, max_width=344)
        self.line("M 1496 528 H 1840", "#E4DCF0", 1)
        for i, label in enumerate(c["optimizer_lines"]):
            self.text(1496, 562 + i * 33, label, 21, max_width=344)

        self.rect(1472, 702, 392, 160)
        self.text(1496, 742, "HiSim", 29, weight=600)
        self.pill(1728, 720, 112, c["standalone"], "#EEF2F7", MUTED, 17)
        self.text(1496, 777, c["hisim"], 22, MUTED)
        for i, label in enumerate(c["hisim_lines"]):
            self.text(1496, 811 + i * 29, label, 19, max_width=344)

        # Storage I/O targets only the actual storage box. EventReport is metadata integration.
        self.rect(560, 914, 1304, 110, "#F0FAF6", "#B9DDD0")
        self.text(584, 951, c["storage_title"], 26, TEAL, 600)
        self.text(584, 987, "TairMemPool (DRAM / SSD) · HF3FS · NFS · Mooncake* · VCNS HF3FS*", 25,
                  max_width=1256)
        self.text(584, 1012, c["optional"], 16, MUTED, max_width=776)

        # Runtime paths. Sparse routing keeps data, metadata and admin traffic distinct.
        self.line("M 420 536 H 488 V 340 H 582", BLUE, 3, "blue")
        self.line("M 420 639 H 488 V 536", BLUE, 3)
        self.line("M 420 727 H 488 V 639", BLUE, 3)
        self.text(500, 282, c["metadata"], 19, BLUE, 600, "middle")
        self.text(500, 310, "gRPC / HTTP", 16, BLUE, anchor="middle")

        self.line("M 1470 340 H 1362", ORANGE, 3, "orange")
        self.text(1416, 307, c["control"], 19, ORANGE, 600, "middle")

        self.line("M 1362 367 H 1416 V 546 H 1470", PURPLE, 2.5, "purple", True)
        self.rect(1388, 442, 56, 58, "#F7F9FC", "none", 0)
        self.text(1416, 464, c["events"], 18, PURPLE, 500, "middle")
        self.text(1416, 492, "gRPC", 16, PURPLE, anchor="middle")

        self.line("M 420 810 H 488 V 968 H 558", TEAL, 5, "teal", both=True)
        self.text(80, 914, c["data"], 23, TEAL, 600)
        self.text(80, 949, c["direct"], 21, TEAL, max_width=368)
        self.line("M 1362 680 H 1416 V 912", "#97A9BE", 2, "gray")
        self.rect(1368, 872, 96, 29, "#F7F9FC", "none", 0)
        self.text(1416, 894, c["backend"], 17, MUTED, 500, "middle")

        # Compact legend and the system-wide reuse constraint.
        for x, key, color, dashed in [(56, "metadata", BLUE, False), (246, "data", TEAL, False),
                                       (402, "control", ORANGE, False), (576, "events", PURPLE, True)]:
            self.line(f"M {x} 1058 h 32", color, 3, dashed=dashed)
            self.text(x + 44, 1065, c[key], 19, MUTED)
        self.text(1864, 1065, c["isolation"], 20, INK, 500, "end", 1050)
        self.parts.extend(["</g>", "</svg>"])
        return "\n".join(self.parts) + "\n"


COMPONENT_COPY = {
    "zh": {
        "subtitle": "服务端组件协作：前台匹配与写入 · 元数据索引 · 后台回收与迁移",
        "entry": "gRPC / HTTP · Leader 门禁 · 组件装配",
        "facade": "系统门面：前台请求编排 + 后台组件协调",
        "config": "config · 配置与 HA",
        "registry": "RegistryManager · 实例 / 配额 / 存储",
        "leader": "LeaderElector · 主备生命周期",
        "foreground": "前台请求路径",
        "searcher_manager": "按 Instance 创建 / 获取 Searcher",
        "searcher": "前缀 / 滑窗 / 混合匹配 · 元数据更新",
        "write": "两阶段写入会话 · 写地址 · 超时清理",
        "selector": "依据配额 / 水位 / 健康选择存储后端",
        "report": "索引更新 / 查询 · Reporter 生命周期与可见性",
        "meta": "meta · 元数据索引",
        "indexer_manager": "按 instance_id 隔离 · 共享 QueryExecutor",
        "indexer": "Cache key → CacheLocation · Get / Put / CAS / Sample",
        "meta_backend": "local / Redis / memory-primary + async Redis · 恢复与备份",
        "storage": "data_storage · 存储后端管理",
        "storage_manager": "URI · 健康 · 水位",
        "physical": "TairMemPool / HF3FS / Mooncake / NFS 等",
        "event_backend": "Reporter / Snapshot · GC 清理判定",
        "background": "后台维护",
        "reclaimer": "Quota / 水位 · Group LRU · 公平回收",
        "gc": "扫描索引 · Backend 批量判定",
        "migration": "Reclaimer 触发 / Admin · Copy / Mark",
        "executor": "共享线程池 · 优先级调度 · 迁移预算",
        "executor_ops": "删除：Get / CAS / Sync → Delete / CAD",
        "supervisor": "删除任务提交与结果跟踪",
        "call": "调用",
        "configuration": "配置",
        "search": "查询 / 更新",
        "sample": "采样 / 扫描",
        "select": "选择后端",
        "via_registry": "经 Registry",
        "submit": "提交异步任务",
        "execute": "索引 / 存储",
        "state": "EventReportBackend 维护推理侧缓存状态；ReportEvent / GC 仅操作元数据",
        "support": ["event · EventManager → Optimizer 事件流", "metrics · 指标 / Prometheus", "protocol / common · 协议 / 通用基础设施"],
        "legend": ["请求 / 索引", "后端访问", "后台任务", "创建 / 协调"],
        "note": "主要运行时协作关系；配置和通用支撑被多组件复用",
    },
    "en": {
        "subtitle": "Server components: foreground matching & writes · metadata indexes · reclaim & migration",
        "entry": "gRPC / HTTP · Leader gate · assembly",
        "facade": "Facade: foreground requests + background orchestration",
        "config": "config · Registry & HA",
        "registry": "RegistryManager · instances / storage",
        "leader": "LeaderElector · leader lifecycle",
        "foreground": "Foreground request paths",
        "searcher_manager": "Create / acquire a Searcher per Instance",
        "searcher": "Prefix / sliding-window / hybrid; updates",
        "write": "Two-phase sessions · locations · expiry",
        "selector": "Select by quota / watermarks / health",
        "report": "Index updates / queries · reporter visibility",
        "meta": "meta · Metadata indexes",
        "indexer_manager": "Isolated by instance_id · shared QueryExecutor",
        "indexer": "Cache key → CacheLocation · Get / Put / CAS / Sample",
        "meta_backend": "local / Redis / memory-primary + async Redis · recovery & backup",
        "storage": "data_storage · Backend management",
        "storage_manager": "URI · health · usage",
        "physical": "TairMemPool / HF3FS / Mooncake / NFS",
        "event_backend": "Reporter / snapshot · GC verdicts",
        "background": "Background maintenance",
        "reclaimer": "Quota / watermarks · Group LRU",
        "gc": "Index scan · batch backend checks",
        "migration": "Reclaimer / Admin · Copy / Mark",
        "executor": "Shared pool · priorities · migration budget",
        "executor_ops": "Delete: Get / CAS / Sync → Delete / CAD",
        "supervisor": "Delete task submission & result tracking",
        "call": "Calls",
        "configuration": "Config",
        "search": "Read / write",
        "sample": "Sample / scan",
        "select": "Select",
        "via_registry": "via Registry",
        "submit": "Submit asynchronous tasks",
        "execute": "Index / I/O",
        "state": "Inference-side cache state; ReportEvent / GC operate on metadata only",
        "support": ["event · EventManager → Optimizer stream", "metrics · counters / Prometheus", "protocol / common · contracts / infrastructure"],
        "legend": ["Requests / indexes", "Backend access", "Background tasks", "Create / coordinate"],
        "note": "Key runtime relationships; configuration and infrastructure are shared",
    },
}


class ComponentDiagram(Diagram):
    """Runtime collaborations; boxes group responsibilities, not strict ownership."""

    def __init__(self, language):
        super().__init__(
            language,
            title="Tair KVCache Manager — server component relationships",
            description=(
                "Service calls CacheManager, which coordinates foreground and background components. "
                "MetaSearcherManager creates MetaSearcher; MetaSearcher reads and updates MetaIndexer. "
                "MetaIndexerManager manages per-instance MetaIndexer objects, which access "
                "MetaStorageBackendManager. DataStorageSelector uses DataStorageManager via RegistryManager. "
                "CacheReclaimer, CacheGarbageCollector and MigrationManager share SchedulePlanExecutor, "
                "which operates on indexes and storage backends. Reclaimer can trigger migration. "
                "Inference services report their cache state through ReportEvent, which updates indexes "
                "via Searcher and maintains EventReportBackend state. EventReportBackend provides reporter "
                "visibility and GC cleanup verdicts. ReportEvent and event-report GC operate on metadata "
                "only. ReclaimerTaskSupervisor submits and tracks delete tasks."
            ),
        )

    def render(self):
        c = COMPONENT_COPY[self.language]
        self.rect(0, 0, 1920, 1080, "#F7F9FC", "none", 0)
        self.rect(56, 51, 7, 78, "#F18536", "none", 3)
        self.text(83, 88, "Tair KVCache Manager", 42, weight=700)
        self.text(84, 132, c["subtitle"], 24, MUTED, max_width=1450)
        self.text(1864, 82, "SERVER COMPONENTS", 18, MUTED, 500, "end")
        self.text(1864, 113, "KVCM  /  RUNTIME COLLABORATION", 16, "#8595A9", anchor="end")

        self.rect(56, 182, 464, 112)
        self.text(80, 223, "service · Server / *ServiceImpl", 25, weight=600, max_width=416)
        self.text(80, 266, c["entry"], 20, MUTED, max_width=416)

        self.rect(616, 182, 740, 112, "url(#manager)", "#B7CDF4")
        self.text(640, 224, "CacheManager", 32, BLUE, 600)
        self.text(640, 266, c["facade"], 22, MUTED, max_width=692)

        self.rect(1456, 182, 408, 112, "#F5F7FC", "#D7DEED")
        self.text(1480, 214, c["config"], 24, weight=600, max_width=360)
        self.text(1480, 248, c["registry"], 18, MUTED, max_width=360)
        self.text(1480, 278, c["leader"], 18, MUTED, max_width=360)

        self.line("M 522 238 H 614", BLUE, 3, "blue")
        self.text(568, 221, c["call"], 18, BLUE, 500, "middle")
        self.line("M 1454 238 H 1358", "#97A9BE", 2, "gray", True)
        self.text(1406, 221, c["configuration"], 18, MUTED, 500, "middle")
        self.line("M 986 294 V 320 H 68 V 906", "#97A9BE", 2, dashed=True)
        self.line("M 986 320 V 342", "#97A9BE", 2, "gray", True)
        self.line("M 986 320 H 1660 V 342", "#97A9BE", 2, "gray", True)

        self.rect(56, 344, 464, 610, "white", "#C9D9F0")
        self.text(80, 383, c["foreground"], 28, BLUE, 600, max_width=416)
        # A CacheManager coordination spine connects every foreground helper.
        # It is drawn after the group background and before the helper cards.
        self.line("M 68 344 V 906", "#97A9BE", 1.5, dashed=True)
        for y in [446, 678, 798, 906]:
            self.line(f"M 68 {y} H 78", "#97A9BE", 1.5, "gray")
        for y, name, key, size in [
            (410, "MetaSearcherManager", "searcher_manager", 26),
            (514, "MetaSearcher", "searcher", 27),
            (634, "WriteLocationManager", "write", 26),
            (754, "DataStorageSelector", "selector", 26),
        ]:
            height = 72 if key == "searcher_manager" else 88
            self.rect(80, y, 416, height, "#F5F8FE", "#D6E1F1", 12)
            self.text(98, y + 31, name, size, weight=600, max_width=380)
            self.text(98, y + height - 17, c[key], 18, MUTED, max_width=380)
        self.line("M 288 482 V 512", BLUE, 2, "blue")
        self.rect(80, 874, 416, 64, "#F0F5FF", "#D2E0FA", 12)
        self.text(98, 899, "ReportEvent / GetHostCacheState", 22, BLUE, 500, max_width=380)
        self.text(98, 925, c["report"], 17, MUTED, max_width=380)

        self.rect(616, 344, 740, 364, "white", "#C9D9F0")
        self.text(640, 383, c["meta"], 28, BLUE, 600, max_width=692)
        self.rect(640, 410, 692, 64, "#F5F8FE", "#D6E1F1", 12)
        self.text(658, 437, "MetaIndexerManager", 26, weight=600)
        self.text(658, 463, c["indexer_manager"], 19, MUTED, max_width=656)
        self.line("M 986 474 V 502", BLUE, 2, "blue")
        self.rect(640, 504, 692, 78, "#EDF4FF", "#B7CDF4", 12)
        self.text(658, 538, "MetaIndexer", 29, BLUE, 600)
        self.text(658, 568, c["indexer"], 20, MUTED, max_width=656)
        self.line("M 986 582 V 610", BLUE, 2, "blue")
        self.rect(640, 612, 692, 72, "#F5F8FE", "#D6E1F1", 12)
        self.text(658, 643, "MetaStorageBackendManager", 25, weight=600)
        self.text(658, 673, c["meta_backend"], 18, MUTED, max_width=656)

        self.rect(616, 744, 740, 226, "white", "#BEDFD4")
        self.text(640, 782, c["storage"], 27, TEAL, 600, max_width=692)
        self.rect(640, 800, 692, 60, "#ECF8F2", "#BBDDD1", 12)
        self.text(658, 837, "DataStorageManager", 27, TEAL, 600)
        self.text(1314, 837, c["storage_manager"], 20, MUTED, anchor="end", max_width=320)
        self.line("M 986 860 V 878 H 816 V 892", TEAL, 2, "teal")
        self.line("M 986 878 H 1174 V 892", TEAL, 2, "teal")
        self.rect(640, 894, 352, 54, "#F4FAF7", "#D0E6DD", 10)
        self.text(658, 918, "DataStorageBackend", 23, weight=600, max_width=316)
        self.text(658, 941, c["physical"], 15, MUTED, max_width=316)
        self.rect(1016, 894, 316, 54, "#F0F5FF", "#D2E0FA", 10)
        self.text(1032, 918, "EventReportBackend", 23, BLUE, 600, max_width=284)
        self.text(1032, 941, c["event_backend"], 16, MUTED, max_width=284)

        self.rect(1456, 344, 408, 364, "#FFFCF7", "#E8D7C3")
        self.text(1480, 383, c["background"], 27, ORANGE, 600, max_width=360)
        for y, name, key, size in [
            (410, "CacheReclaimer", "reclaimer", 27),
            (514, "CacheGarbageCollector", "gc", 25),
            (618, "MigrationManager", "migration", 26),
        ]:
            height = 66 if key == "migration" else 78
            self.rect(1480, y, 360, height, "white", "#E7DCCD", 12)
            self.text(1498, y + 31, name, size, weight=600, max_width=324)
            self.text(1498, y + height - 15, c[key], 18, MUTED, max_width=324)
        # Reclaimer may initiate migration; a narrow route avoids crossing GC.
        self.line("M 1478 450 H 1466 V 650 H 1478", ORANGE, 2, "orange")

        self.rect(1456, 780, 408, 190, "#FFF9EF", "#E4CDB1")
        self.text(1480, 816, "SchedulePlanExecutor", 27, ORANGE, 600, max_width=360)
        self.text(1480, 851, c["executor"], 18, MUTED, max_width=360)
        self.text(1480, 881, c["executor_ops"], 17, MUTED, max_width=360)
        self.line("M 1480 900 H 1840", "#E7D8C4", 1)
        self.text(1480, 926, "ReclaimerTaskSupervisor", 22, weight=600, max_width=360)
        self.text(1480, 952, c["supervisor"], 17, MUTED, max_width=360)

        # Major runtime calls. The diagram intentionally omits secondary edges.
        self.line("M 498 558 H 638", BLUE, 3, "blue")
        self.text(566, 538, c["search"], 17, BLUE, 500, "middle")
        self.line("M 498 798 H 556 V 830 H 638", TEAL, 3, "teal")
        self.text(558, 777, c["select"], 17, TEAL, 500, "middle")
        self.text(558, 858, c["via_registry"], 15, MUTED, anchor="middle")
        self.line("M 498 908 H 580 V 978 H 1174 V 950", BLUE, 2.5, "blue")
        self.text(640, 1000, c["state"], 17, MUTED, max_width=700)

        self.line("M 1454 552 H 1334", "#97A9BE", 2, "gray", True)
        self.text(1394, 532, c["sample"], 16, MUTED, anchor="middle")
        self.line("M 1842 450 H 1888 V 756 H 1660 V 778", ORANGE, 3, "orange")
        self.line("M 1842 552 H 1888", ORANGE, 3)
        self.line("M 1842 650 H 1888", ORANGE, 3)
        self.text(1660, 744, c["submit"], 19, ORANGE, 500, "middle")

        self.line("M 1454 864 H 1384 V 578 H 1334", ORANGE, 2.5, "orange")
        self.line("M 1384 830 H 1334", ORANGE, 2.5, "orange")
        self.rect(1359, 716, 74, 25, "#F7F9FC", "none", 0)
        self.text(1394, 735, c["execute"], 15, ORANGE, 500, "middle")

        self.rect(56, 1012, 1808, 34, "#ECF1F7", "none", 8)
        for x, value, width in zip([76, 778, 1250], c["support"], [680, 450, 590]):
            self.text(x, 1036, value, 19, MUTED, max_width=width)
        positions = [56, 260, 454, 660] if self.language == "zh" else [56, 300, 516, 756]
        for x, label, color, dashed in zip(
            positions, c["legend"], [BLUE, TEAL, ORANGE, "#97A9BE"], [False, False, False, True]
        ):
            self.line(f"M {x} 1065 h 26", color, 2.5, dashed=dashed)
            self.text(x + 36, 1072, label, 17, MUTED)
        self.text(1864, 1072, c["note"], 16, MUTED, anchor="end", max_width=780)
        self.parts.extend(["</g>", "</svg>"])
        return "\n".join(self.parts) + "\n"


DEPLOYMENT_COPY = {
    "zh": {
        "subtitle": "部署视图：推理服务 · KVCM 一主多备 · 共享 KVCache 存储",
        "inference": "推理服务",
        "same_instance": "同一 Instance",
        "node": "推理节点",
        "memory": "GPU / 主机内存",
        "client": "Client / Connector（随推理进程部署）",
        "metadata": "Meta / ReportEvent",
        "metadata_plane": "元数据面",
        "metadata_ops": "查询 / 两阶段写入 / ReportEvent",
        "discovery": "GetClusterInfo 发现 Leader 后直连",
        "cluster": "Tair KVCache Manager 服务",
        "cluster_note": "一主多备 · 客户端自动发现 Leader 与切主重连",
        "leader": "元数据 / 管控业务读写",
        "leader_match": "匹配 · 写入确认 · ReportEvent",
        "leader_background": "配额回收 · GC · 分层迁移",
        "manager_note": "统一管理 KVCache 的位置、状态与容量",
        "persistence": "持久化存储",
        "persistence_desc": "配置 / 元数据 · 可选 Redis 等后端",
        "backend_ops": "Leader 后端管理",
        "backend_mutations": "创建 / 删除 / 迁移",
        "backend_health": "健康检查 / 水位",
        "direct": "KVCache 数据直连",
        "direct_desc": "高带宽数据读写",
        "storage": "存储集群",
        "storage_note": "可插拔后端 · 容量按需扩展",
        "filesystem": "文件存储后端",
        "optional": "* 可选构建后端",
        "legend": ["元数据请求", "数据直连", "后端管理 / 持久化"],
        "scope": "复用边界：相同 instance_id 内，跨请求 / 跨节点",
    },
    "en": {
        "subtitle": "Deployment: inference services · KVCM active-standby · shared KVCache storage",
        "inference": "Inference services",
        "same_instance": "Same Instance",
        "node": "Inference node",
        "memory": "GPU / host memory",
        "client": "Client / Connector (in each inference process)",
        "metadata": "Meta / ReportEvent",
        "metadata_plane": "Metadata plane",
        "metadata_ops": "Query / two-phase writes / ReportEvent",
        "discovery": "Discover the Leader via GetClusterInfo",
        "cluster": "Tair KVCache Manager service",
        "cluster_note": "Active-standby · leader discovery & failover routing",
        "leader": "Metadata / control requests",
        "leader_match": "Match · confirm writes · ReportEvent",
        "leader_background": "Quota & reclaim · GC · tiered migration",
        "manager_note": "Global KVCache locations, state and capacity",
        "persistence": "Persistent storage",
        "persistence_desc": "Configuration / metadata · e.g. Redis (optional)",
        "backend_ops": "Leader backend ops",
        "backend_mutations": "Create / delete / migrate",
        "backend_health": "Health / watermarks",
        "direct": "Direct KVCache I/O",
        "direct_desc": "High-bandwidth payload I/O",
        "storage": "Storage cluster",
        "storage_note": "Pluggable backends · scale capacity as needed",
        "filesystem": "File storage backends",
        "optional": "* Optional build-time backends",
        "legend": ["Metadata", "Direct I/O", "Backend ops / persistence"],
        "scope": "Reuse scope: same instance_id, across requests / nodes",
    },
}


class DeploymentDiagram(Diagram):
    """Three primary deployment domains, with persistence as an auxiliary dependency."""

    def __init__(self, language):
        super().__init__(
            language,
            title="Tair KVCache Manager — deployment architecture",
            description=(
                "Three primary domains: inference services, the active-standby KVCM service, "
                "and shared KVCache storage. Client/Connector runs in each inference process. "
                "Clients discover the Leader through GetClusterInfo, then send metadata requests "
                "and inference-side ReportEvent state reports to the Leader. TransferClient reads "
                "and writes payloads directly to shared storage. The Leader manages storage health, "
                "allocation, deletion and tiered migration. Followers recover state before serving "
                "business requests after promotion. Persistent configuration and metadata storage "
                "is an auxiliary dependency, with Redis as an optional implementation. Cache reuse "
                "is isolated by instance_id."
            ),
        )

    def render(self):
        c = DEPLOYMENT_COPY[self.language]
        self.rect(0, 0, 1920, 1080, "#F7F9FC", "none", 0)
        self.rect(56, 57, 7, 78, "#F18536", "none", 3)
        self.text(83, 93, "Tair KVCache Manager", 44, weight=700)
        self.text(84, 138, c["subtitle"], 25, MUTED, max_width=1640)
        self.text(1864, 88, "DEPLOYMENT ARCHITECTURE", 18, MUTED, 500, "end")

        # The three primary groups follow the original deployment diagram.
        self.rect(360, 192, 1200, 280, "white", "#C9D9F0", 20)
        self.text(384, 235, c["inference"], 32, BLUE, 600, max_width=860)
        self.pill(1376, 210, 160, c["same_instance"], "#EAF0FB", BLUE, 18)
        self.text(384, 274, "vLLM · SGLang · RTP-LLM · TRT-LLM", 24, MUTED, max_width=1152)
        for x, node in [(384, "A"), (976, "B")]:
            self.rect(x, 300, 560, 148, "#FAFBFE", "#D8E2EF", 14)
            self.text(x + 20, 335, f'{c["node"]} {node} · {c["memory"]}',
                      25, weight=600, max_width=520)
            self.text(x + 20, 370, c["client"], 21, MUTED, max_width=520)
            self.rect(x + 20, 388, 242, 42, "#F0F5FF", "#B7CDF4", 10)
            self.text(x + 141, 416, c["metadata"], 21, BLUE, 600, "middle", 210)
            self.rect(x + 278, 388, 262, 42, "#EAF8F3", "#BBE2D7", 10)
            self.text(x + 409, 416, "TransferClient", 23, TEAL, 600, "middle", 230)

        self.rect(56, 588, 864, 352, "white", "#B9CFEA", 20, 2)
        self.text(80, 632, c["cluster"], 32, BLUE, 600, max_width=816)
        self.text(80, 666, c["cluster_note"], 21, MUTED, max_width=816)

        # Standby instances sit behind the Leader and carry only a compact label.
        self.rect(128, 722, 768, 166, "#EDF1F6", "#D9E1EC", 14)
        self.rect(104, 704, 768, 166, "#F4F6FA", "#D9E1EC", 14)
        self.rect(80, 686, 768, 166, "url(#manager)", "#B7CDF4", 14)
        self.text(104, 727, "KVCM · Leader", 31, BLUE, 600, max_width=720)
        self.text(104, 771, c["leader"], 23, INK, 500, max_width=720)
        self.text(104, 817, c["leader_match"], 18, MUTED, max_width=336)
        self.line("M 456 797 V 828", "#D4E0F3", 1.5)
        self.text(472, 817, c["leader_background"], 18, MUTED, max_width=352)
        self.text(872, 879, "Follower × N", 17, "#8190A6", anchor="end", max_width=220)
        self.text(80, 917, c["manager_note"], 22, BLUE, 500, max_width=816)

        self.rect(1136, 588, 728, 440, "white", "#B9DDD0", 20, 2)
        self.text(1160, 632, c["storage"], 32, TEAL, 600, max_width=680)
        self.text(1160, 666, c["storage_note"], 21, MUTED, max_width=680)
        for y, name, detail in [
            (694, "TairMemPool", "DRAM / SSD"),
            (804, "HF3FS · NFS", c["filesystem"]),
            (914, "Mooncake* · VCNS HF3FS*", c["optional"]),
        ]:
            self.rect(1160, y, 680, 86, "#F1FAF5", "#D0E6DD", 12)
            self.text(1184, y + 35, name, 28, weight=600, max_width=632)
            self.text(1184, y + 67, detail, 21, MUTED, max_width=632)

        # Persistence is deliberately secondary in the visual hierarchy.
        self.rect(224, 966, 528, 62, "#EFF3F8", "#DFE5EF", 12, 1)
        self.text(488, 991, c["persistence"], 22, MUTED, 500, "middle", 480)
        self.text(488, 1017, c["persistence_desc"], 17, MUTED, anchor="middle", max_width=480)
        self.line("M 488 942 V 964", "#97A9BE", 1.8, "gray", True)

        # Inference drives both planes; payload I/O takes the direct storage path.
        self.line("M 600 474 V 520 H 488 V 586", BLUE, 3, "blue")
        self.text(80, 505, c["metadata_plane"], 25, BLUE, 600, max_width=360)
        self.text(80, 538, c["metadata_ops"], 18, BLUE, max_width=360)
        self.text(80, 567, c["discovery"], 17, MUTED, max_width=360)
        self.line("M 1320 474 V 520 H 1500 V 586", TEAL, 6, "teal", both=True)
        self.text(1580, 505, c["direct"], 24, TEAL, 600, max_width=284)
        self.text(1580, 540, c["direct_desc"], 19, TEAL, max_width=284)

        self.line("M 922 790 H 1134", "#97A9BE", 2.5, "gray", both=True)
        self.text(1028, 739, c["backend_ops"], 21, MUTED, 500, "middle", 204)
        self.text(1028, 772, c["backend_mutations"], 17, MUTED, anchor="middle", max_width=204)
        self.text(1028, 825, c["backend_health"], 17, MUTED, anchor="middle", max_width=204)

        for x, label, color, dashed in zip(
            [56, 266, 464], c["legend"], [BLUE, TEAL, "#97A9BE"], [False, False, False]
        ):
            self.line(f"M {x} 1058 h 28", color, 3, dashed=dashed)
            self.text(x + 40, 1065, label, 18, MUTED)
        self.text(1864, 1065, c["scope"], 19, INK, 500, "end", max_width=800)
        self.parts.extend(["</g>", "</svg>"])
        return "\n".join(self.parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-only", action="store_true", help="Skip optional PNG rendering")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    renderer = shutil.which("rsvg-convert")
    for stem, diagram in [("kvcm-architecture-overview", Diagram),
                          ("kvcm-server-components", ComponentDiagram),
                          ("kvcm-deployment", DeploymentDiagram)]:
        for language in COPY:
            path = output / f"{stem}-{language}.svg"
            path.write_text(diagram(language).render(), encoding="utf-8")
            print(path)
            if renderer and not args.svg_only:
                png_path = path.with_suffix(".png")
                subprocess.run([renderer, "--width", "3840", "--height", "2160",
                                "--output", str(png_path), str(path)], check=True)
                print(png_path)
    if not renderer and not args.svg_only:
        print("PNG export skipped: install rsvg-convert, then run again.")


if __name__ == "__main__":
    main()
