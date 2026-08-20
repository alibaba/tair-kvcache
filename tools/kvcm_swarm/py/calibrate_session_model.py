#!/usr/bin/env python3
"""会话模型离线标定器。

纯离线：按 docs/design/kvcm_swarm.md §6 的会话模型生成合成请求流，统计 keyspace
形态并与 accesslogs/ 实测值对比。不需要 kvcm 服务。

目的：在写 C++ 引擎之前验证会话类参数能否复现生产的复用结构。
"""
import argparse, hashlib, itertools, json, random, statistics
from collections import defaultdict

# ---- 对比基线 ----
# 基线值来自生产实测，属内部数据，不入库。用 --baseline <json> 传入。
# 格式: {"<指标名>": [<基线值>, "<中文描述>"], ...}
# 内容与配套约束见 docs/design/kvcm_swarm_workload_data.md（内部）。
# 不传 --baseline 时只输出合成流量自身的形态统计，不做对比。
TARGETS: dict = {}


def load_baseline(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {k: (float(v[0]), v[1]) for k, v in raw.items()}


DEFAULT_REQUESTS = 5505   # 复用统计随窗口增长，比较时必须对齐等价"人均历史"


def sample(d, rng):
    if "const" in d:   return d["const"]
    if "uniform" in d: lo, hi = d["uniform"]; return rng.randint(lo, hi)
    if "discrete" in d:
        ks = list(d["discrete"])
        return int(rng.choices(ks, weights=[d["discrete"][k] for k in ks])[0])
    if "mixture" in d:
        c = rng.choices(d["mixture"], weights=[x["w"] for x in d["mixture"]])[0]
        return sample({k: v for k, v in c.items() if k != "w"}, rng)
    raise ValueError(d)


class PrefixPool:
    """共享前缀池。root 是一串固定 slot content id，可被多会话（多客户端）挂载。"""
    def __init__(self, name, roots, root_blocks, rng, zipf=0.0):
        self.roots = [[f"{name}/r{i}/s{j}" for j in range(sample(root_blocks, rng))]
                      for i in range(roots)]
        self.weights = [1.0 / (i + 1) ** zipf for i in range(roots)] if zipf else None

    def pick(self, rng):
        return (rng.choices(self.roots, weights=self.weights)[0]
                if self.weights else rng.choice(self.roots))


_sid = itertools.count(1)


class Session:
    """会话：一等概念。有明确轮数，结束即可从 shadow state 移除。"""
    def __init__(self, cls, pools, rng):
        self.sid = next(_sid)
        self.cls = cls
        self.turns_left = sample(cls["turns"], rng)
        self.total_turns = self.turns_left
        self.client = None          # 由 LB 决定，可跨轮变化
        self.seen = {}              # client_id -> 该客户端本地已持有的前缀长度（模型 local_miss_keys）
        self.slots = []
        self._pseq = 0
        L1 = sample(cls["first_turn_blocks"], rng)
        root = (pools[cls["root_sharing"]["pool"]].pick(rng)
                if rng.random() < cls["root_sharing"]["shared_prob"] else [])
        self.slots = [root[i] if i < len(root) else self._priv() for i in range(L1)]

    def _priv(self):
        self._pseq += 1
        return f"s{self.sid}/p{self._pseq}"

    def advance(self, rng):
        """推进一轮：末尾 tail_not_reused 个 slot 换新身份，再追加 blocks_per_turn 个。
        返回本轮"新增+换身份"的 slot 数，用于计算本地缺失范围。"""
        tail = sample(self.cls["tail_not_reused"], rng)
        for i in range(max(0, len(self.slots) - tail), len(self.slots)):
            self.slots[i] = self._priv()
        grew = sample(self.cls["blocks_per_turn"], rng)
        for _ in range(grew):
            self.slots.append(self._priv())
        return tail + grew


def sec_key(k):
    """次组 (mamba) 的 key：与主组同 slot 但不同 spec，故 key 不同。"""
    return k ^ 0x5bf0_3635_1a2b_4c7d


def block_keys(slots, pool, cache):
    """前缀链派生 → 复刻 V6D 的 int64(sha256(s)[:8], signed)。"""
    keys, chain = [], 0
    for s in slots:
        ck = (chain, s)
        v = cache.get(ck)
        if v is None:
            c2 = int.from_bytes(hashlib.blake2b(f"{chain:016x}/{s}".encode(),
                                                digest_size=8).digest(), "big")
            k = int.from_bytes(hashlib.sha256(f"{pool}/{c2:016x}".encode()).digest()[:8],
                               "big", signed=True)
            v = cache[ck] = (c2, k)
        chain, k = v
        keys.append(k)
    return keys


def run(cfg, n_clients, n_requests, seed):
    rng = random.Random(seed)
    pools = {p["name"]: PrefixPool(p["name"], p["roots"], p["root_blocks"], rng,
                                   p.get("zipf", 0.0))
             for p in cfg["prefix_pools"]}
    classes, weights = cfg["session_classes"], [c["weight"] for c in cfg["session_classes"]]
    affinity = cfg["session_affinity"]          # 同一会话下一轮留在同客户端的概率
    scope = cfg["lookup_scope"]
    n_active = int(n_clients * cfg.get("active_sessions_per_client", 1))
    cache = {}

    def new_session():
        return Session(rng.choices(classes, weights=weights)[0], pools, rng)

    active = [new_session() for _ in range(n_active)]
    lookups, n_writes = [], 0
    key_clients, key_count = defaultdict(set), defaultdict(int)
    reported, hb, mb = set(), 0, 0
    kind = defaultdict(int)
    fin_turns, fin_blocks = [], []

    while len(lookups) + n_writes < n_requests:
        idx = rng.randrange(n_active)
        s = active[idx]

        # LB 路由：以 affinity 概率沿用上一轮客户端，否则重新路由
        if s.client is None or rng.random() >= affinity:
            s.client = rng.randrange(n_clients)
        cid = s.client

        if s.turns_left < s.total_turns:        # 非首轮才推进
            tail_changed = s.advance(rng)
        else:
            tail_changed = len(s.slots)         # 首轮：整段都是新的
        s.turns_left -= 1

        ks = block_keys(s.slots, s.cls["root_sharing"]["pool"], cache)
        L = len(ks)

        # 本地缺失范围 (peer.py:1551 local_miss_keys)
        seen = s.seen.get(cid, 0)
        lo = 0 if seen == 0 else max(0, min(seen, L - tail_changed, L))
        missing = list(range(lo, L))
        s.seen[cid] = L
        if scope.get("local_evict_prob", 0.0) and rng.random() < scope["local_evict_prob"]:
            s.seen[cid] = 0                          # 本地被淘汰 → 下次全量重查

        # 每个推理请求发多次 get()：主组(全注意力, 大) + 稀疏次组(mamba, 小)
        # 依据实测同 trace_id 的两次 lookup 长度对极不均等: (356,3) (331,45) (3,15)
        stride = cfg.get("secondary_stride", 0)
        groups = [[ks[i] for i in missing]]
        if stride:
            groups.append([sec_key(ks[i]) for i in missing if i % stride == 0])
        for q in groups:
            if not q:
                continue
            lookups.append(q)
            for k in q:
                key_count[k] += 1
                key_clients[k].add(cid)
            h = sum(1 for k in q if k in reported)
            hb += h; mb += len(q) - h
            kind["full" if h == len(q) else "zero" if h == 0 else "partial"] += 1
            reported.update(q)

        # 淘汰溢写：生产 StartWriteCache/lookup = 1.40，均值 85.7 block，硬顶 128
        n_spill = int(rng.random() < (cfg["spill_rate"] % 1)) + int(cfg["spill_rate"])
        for _ in range(n_spill):
            if L == 0: break
            n = min(128, L)
            span = max(1, L - n + 1)
            bias = cfg.get("spill_front_bias", 0.0)
            off = int(rng.random() ** (1 + 3 * bias) * span)   # bias 越大越偏前(最冷)
            w = ks[off:off + n]                       # 连续窗口，模拟批量 LRU 淘汰
            n_writes += 1
            for k in w:
                key_count[k] += 1
                key_clients[k].add(cid)

        if s.turns_left <= 0:                   # 会话结束 → 回收，补新会话
            fin_turns.append(s.total_turns); fin_blocks.append(len(s.slots))
            active[idx] = new_session()

    mentions, distinct = sum(key_count.values()), len(key_count)
    lk = sorted(len(q) for q in lookups)
    pct = lambda xs, p: xs[min(len(xs) - 1, int(len(xs) * p / 100))] if xs else 0

    by_first = defaultdict(list)
    for q in lookups:
        if q: by_first[q[0]].append(q)
    lcps, strict = [], 0
    for grp in by_first.values():
        for a, b in zip(grp, grp[1:]):
            n = 0
            for x, y in zip(a, b):
                if x != y: break
                n += 1
            lcps.append(n)
            strict += (n == min(len(a), len(b)))

    return {
        "reuse_mean": mentions / max(distinct, 1),
        "reuse_gt1_pct": 100 * sum(v > 1 for v in key_count.values()) / max(distinct, 1),
        "_reuse_max": max(key_count.values(), default=0),   # 外延量, 非标定目标
        "lookup_p50": pct(lk, 50), "lookup_p90": pct(lk, 90), "lookup_max": lk[-1] if lk else 0,
        "lcp_median": statistics.median(lcps) if lcps else 0,
        "lcp_mean": statistics.fmean(lcps) if lcps else 0,
        "lcp_max": max(lcps, default=0),
        "strict_prefix_pct": 100 * strict / max(len(lcps), 1),
        "cross_client_pct": 100 * sum(len(v) > 1 for v in key_clients.values()) / max(len(key_clients), 1),
        "_hit_upper": 100 * hb / max(hb + mb, 1),
        "_req": (100 * kind["full"] / len(lookups), 100 * kind["zero"] / len(lookups),
                 100 * kind["partial"] / len(lookups)),
        "_pairs": len(lcps), "_fin": len(fin_turns),
        "_turns": statistics.fmean(fin_turns) if fin_turns else 0,
        "_sblocks": statistics.fmean(fin_blocks) if fin_blocks else 0,
    }


def score(got):
    """平均相对误差，越小越好。无基线时返回 None。"""
    devs = [abs(got[k] - t) / t for k, (t, _) in TARGETS.items() if t and k in got]
    return statistics.fmean(devs) if devs else None


SHAPE_KEYS = ["reuse_mean", "reuse_gt1_pct", "lookup_p50", "lookup_p90", "lookup_max",
              "lcp_median", "lcp_mean", "lcp_max", "strict_prefix_pct", "cross_client_pct"]


def report(name, got):
    sc = score(got)
    tail = f"   [平均相对误差 {sc*100:.0f}%]" if sc is not None else "   [无基线，仅输出形态]"
    print(f"\n{'='*76}\n{name}{tail}\n{'='*76}")
    if TARGETS:
        print(f"{'指标':<26} {'合成':>10} {'基线':>10} {'偏差':>9}")
        print("-" * 76)
        for k, (tgt, desc) in TARGETS.items():
            if k not in got:
                continue
            g, dev = got[k], (got[k] - tgt) / tgt * 100
            flag = "  " if abs(dev) <= 30 else (" ~" if abs(dev) <= 100 else "!!")
            print(f"{desc:<26} {g:>10.1f} {tgt:>10.1f} {dev:>+8.0f}% {flag}")
    else:
        print(f"{'指标':<26} {'合成':>10}")
        print("-" * 76)
        for k in SHAPE_KEYS:
            print(f"{k:<26} {got[k]:>10.1f}")
    print("-" * 76)
    print(f"单key最高复用={got['_reuse_max']}(外延量,随流量增长,不作标定目标) "
          f"完成会话={got['_fin']} 均值轮数={got['_turns']:.2f} "
          f"均值block/会话={got['_sblocks']:.0f} LCP配对={got['_pairs']}")
    print(f"命中率上界(无限缓存)={got['_hit_upper']:.1f}  "
          f"请求级全/零/部分={got['_req'][0]:.1f}/{got['_req'][1]:.1f}/{got['_req'][2]:.1f}")


BASE = {
    "session_affinity": 0.80,
    "active_sessions_per_client": 20,
    "spill_rate": 1.40,   # StartWriteCache / lookup, 实测
    "secondary_stride": 64,   # 次组稀疏步长(0=不发次组)
    "spill_front_bias": 1.0,  # 溢写窗口偏向序列前部的程度
    "lookup_scope": {"local_evict_prob": 0.35},
    "session_classes": [
        {"name": "single_turn", "weight": 0.35, "turns": {"const": 1},
         "first_turn_blocks": {"mixture": [{"w": 0.7, "uniform": [4, 40]},
                                          {"w": 0.3, "uniform": [200, 1300]}]},
         "blocks_per_turn": {"const": 0}, "tail_not_reused": {"const": 0},
         "root_sharing": {"shared_prob": 0.40, "pool": "sys_prompt_pool"}},
        {"name": "short_chat", "weight": 0.45, "turns": {"uniform": [2, 6]},
         "first_turn_blocks": {"uniform": [4, 60]}, "blocks_per_turn": {"uniform": [2, 20]},
         "tail_not_reused": {"discrete": {0: 0.63, 1: 0.30, 2: 0.05, 3: 0.02}},
         "root_sharing": {"shared_prob": 0.50, "pool": "sys_prompt_pool"}},
        {"name": "long_context", "weight": 0.20, "turns": {"uniform": [6, 30]},
         "first_turn_blocks": {"uniform": [200, 1300]}, "blocks_per_turn": {"uniform": [5, 40]},
         "tail_not_reused": {"discrete": {0: 0.63, 1: 0.30, 2: 0.05, 3: 0.02}},
         "root_sharing": {"shared_prob": 0.30, "pool": "doc_pool"}},
    ],
    "prefix_pools": [
        {"name": "sys_prompt_pool", "roots": 40, "root_blocks": {"uniform": [2, 12]}, "zipf": 1.0},
        {"name": "doc_pool", "roots": 150, "root_blocks": {"uniform": [50, 400]}, "zipf": 1.0},
    ],
}


def variant(**over):
    import copy
    c = copy.deepcopy(BASE)
    for k, v in over.items():
        if k == "local_evict":       c["lookup_scope"]["local_evict_prob"] = v
        elif k == "zipf":            [p.update(zipf=v) for p in c["prefix_pools"]]
        elif k == "pool_roots":      c["prefix_pools"][0]["roots"], c["prefix_pools"][1]["roots"] = v
        elif k == "sess_per_client":  c["active_sessions_per_client"] = v
        elif k == "spill_rate":       c["spill_rate"] = v
        elif k == "sec_stride":       c["secondary_stride"] = v
        elif k == "front_bias":       c["spill_front_bias"] = v
        else:                        c[k] = v
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    ap.add_argument("--clients", type=int, default=9,
                    help="节点数；与基线比较时应使用与基线相同的规模")
    ap.add_argument("--baseline", type=str, default=None,
                    help="对比基线 json（内部数据，不入库）；不传则只输出形态统计")
    ap.add_argument("--search", action="store_true")
    a = ap.parse_args()
    if a.baseline:
        TARGETS.update(load_baseline(a.baseline))

    if not a.search:
        report(f"当前参数 / {a.clients} 节点", run(BASE, a.clients, a.requests, a.seed))
    else:
        base = dict(zipf=1.0, pool_roots=(40, 150), session_affinity=0.8,
                    sess_per_client=20, front_bias=1.0, sec_stride=64)
        cands = [("H%d G4 + 本地淘汰%.2f" % (i, p), variant(**base, local_evict=p))
                 for i, p in enumerate([0.0, 0.10, 0.20, 0.35, 0.50, 0.70], 1)]
        best = None
        for name, cfg in cands:
            g = run(cfg, a.clients, a.requests, a.seed)
            report(f"{name} / {a.clients} 节点", g)
            if best is None or (score(g) or 1e9) < (score(best[1]) or 1e9): best = (name, g)
        if score(best[1]) is not None:
            print(f"\n>>> 最优: {best[0]}  平均相对误差 {score(best[1])*100:.0f}%")
