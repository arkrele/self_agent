"""JitRL 的纯函数部分：不依赖网络、不依赖数据库，离线可反复验证。

Step 1: score_arms()   —— 优势估计 + logit 修正
Step 2: extract_arms() —— 从 Responses API 的 logprobs 里抽出 N 个候选臂的分布
动作签名: action_sig() / parse_sig() —— (tool, args) 与匹配键之间的双向转换
"""
import json

# ---------------- 超参 ----------------
N_ARMS        = 3      # 每轮提出几个候选动作
BETA          = 3.0    # 优势强度，单位是"logprob 跨度的比例"（见 score_arms 注释）
Z_MEM_PENALTY = 1.0    # 记忆注入臂相对"最差模型臂"再低多少个 logprob 跨度

# 输出协议的哨兵词。提示词要求模型以 "CHOICE: <k>" 结尾，解析时按这个串定位。
CHOICE_SENTINEL = "CHOICE"


# ============ 动作签名： (tool, args) ⇄ 匹配键 ============

def action_sig(tool: str, args: dict | None = None) -> str:
    """把 (工具名, 参数) 规范化成一个**确定性且可逆**的匹配键。

    这个串同时扮演两个角色：
        · texts[arm_id] 的值      —— 交给 score_arms 做匹配
        · 数据库里存的 action 字段 —— 检索结果的 q 的键
    两边必须**逐字符相同**。否则同一语义的动作会被当成两个 arm 互相竞争，
    记忆里的那个 arm 永远匹配不上模型的候选，Â 恒为 0，记忆静默失效。

    **可逆是硬需求**：记忆注入臂的编号 > N_ARMS，没有对应的 LLM 候选可以查
    （cands[arm_id-1] 会越界），只能从签名还原出 (tool, args) 才能真正执行。

    三条实现约束：
      · sort_keys=True  —— 字段顺序不同也得到同一个串
      · 紧凑分隔符(',',':') —— 空白稳定，且比默认的 ", " / ": " 更短
      · **不折叠空白**  —— run_bash 的 command 里连续空格是有意义的，
                           "echo  a" 折叠成 "echo a" 会改变命令语义
    """
    name = tool.strip().lower()
    payload = json.dumps(args or {}, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return f"{name}({payload})"


def parse_sig(sig: str):
    """从签名还原 (tool, args)；解析不了返回 None。

    记忆注入臂必须靠这个才能真正执行。返回 None 而不是抛异常，
    是为了让调用方可以"跳过这条记忆"而不是整轮崩掉。
    """
    if not isinstance(sig, str):
        return None
    name, sep, rest = sig.partition("(")
    if not sep or not rest.endswith(")"):
        return None
    try:
        args = json.loads(rest[:-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return name, args


def score_arms(arm_lps: dict[int, float],
               texts: dict[int, str],
               q: dict[str, list[float]],
               beta: float = BETA,
               z_mem_penalty: float = Z_MEM_PENALTY):
    """JitRL 的核心：把记忆里的优势加到 LLM 的 logprob 上，返回每个 arm 的最终分数。

    参数
        arm_lps : {arm_id: logprob}        LLM 在 CHOICE 位置上的分布
        texts   : {arm_id: action 签名}    arm 对应的动作。签名必须和库里存的完全一致，
                                           否则同一语义的动作会被当成两个 arm 竞争。
        q       : {action: [G, ...]}       记忆召回结果按动作聚合后的 return-to-go
        beta    : 优势强度，单位是"logprob 跨度的比例"

    返回 (scores, debug)
        scores : {arm_id: z'}              直接 max 就是最终选择
        debug  : {V, A, span, injected}    供日志排查

    两个关键决定：

    (1) 在 log 空间做加法（论文式 10），而不是原实现的 exp(logprob) + Â。
        API 返回的 logprob = logit − logZ，logZ 对同一位置的所有臂是同一个常数，
        差值完全干净、自动消掉。概率空间 exp(logprob) 则被 logZ 缩放，和量级固定的
        优势项相加时相对权重会漂移，β 就失去了物理含义。

    (2) 把 logprob 按模型臂的跨度线性归一化到 [-1, 0]。
        原始 logprob 的绝对跨度随模型/温度/候选差异变化很大（实测 6~16 nats）。
        若不归一化，β 必须大到接近跨度才能翻转排序，换模型就得重调。归一化后 β 的
        语义固定为"优势最多能翻转 logprob 跨度的多大比例"，跨模型可移植。

    ⚠️ 优势是相对基线定义的：q 里只有**一个**不同动作时，它自己就是基线，Â ≡ 0，
       修正完全不发生。这直接约束了检索半径和 state 摘要的粒度。
    """
    if not arm_lps:
        raise ValueError("arm_lps 为空：logprob 提取失败时不应该调用 score_arms")

    # ① 归一化模型分布。减最大值只是平移（不改 argmax），再除以跨度让 β 无量纲
    lo, hi = min(arm_lps.values()), max(arm_lps.values())
    span = (hi - lo) or 1.0
    z = {i: (v - hi) / span for i, v in arm_lps.items()}          # ∈ [-1, 0]
    t = dict(texts)

    # ② 候选增广：记忆里平均回报为正、但 LLM 没提出的动作，以地板分注入
    injected = []
    if q:
        floor = -1.0 - z_mem_penalty
        nxt = max(z) + 1
        known = set(t.values())
        for action, gs in q.items():
            if action not in known and sum(gs) / len(gs) > 0:
                z[nxt] = floor
                t[nxt] = action
                injected.append(action)
                nxt += 1

    if not q:
        return z, {"V": 0.0, "A": {}, "span": span, "injected": []}

    # ③ Q̂ / V̂ / Â
    all_g = [g for gs in q.values() for g in gs]
    V = sum(all_g) / len(all_g)                     # 论文口径：只在记忆条目上求均值
    qbar = {a: sum(gs) / len(gs) for a, gs in q.items()}
    A = {a: qbar[a] - V for a in qbar}
    scale = max((abs(x) for x in A.values()), default=0.0) or 1.0
    A = {a: x / scale for a, x in A.items()}        # 对称归一化，Â ∈ [-1, 1]

    scores = {i: z[i] + beta * A.get(t[i], 0.0) for i in z}
    return scores, {"V": V, "A": A, "span": span, "injected": injected}


# ============ Step 2：从 logprobs 里抽出 arm 分布 ============

def parse_arm(token: str, n_arms: int):
    """把 token 文本解析成 arm 编号；不是合法的候选编号就返回 None。

    两道过滤都是实测踩出来的，缺一个就会静默出错：

      · isascii() 必须。
        str.isdigit() 对 '①'（带圈数字）'２'（全角）'۲'（阿拉伯-印度数字）都返回 True，
        但 int('①') 直接抛 ValueError。

      · s == str(int(s)) 必须。
        否则 int('01') == 1，于是 ('01', -14.26) 会**静默覆盖** ('1', -0.16)，
        臂 1 从最优变成最差，而整个过程不报任何错。
    """
    s = token.strip()
    if not (s.isascii() and s.isdigit()):
        return None
    if s != str(int(s)):                       # 拒 '01' / '007'
        return None
    v = int(s)
    return v if 1 <= v <= n_arms else None


def collect_logprobs(resp):
    """从 Responses API 的响应里取出 output_text 部分的 logprobs。

    只扫 output_text —— deepseek-flash 是推理模型，会先输出一个 reasoning item，
    那里的 reasoning_text 不是我们要的 arm 分布。

    返回 list[Logprob]；没拿到返回 None。
    """
    for item in resp.output:
        if getattr(item, "type", None) != "message":
            continue
        for part in (item.content or []):
            if getattr(part, "type", None) == "output_text" and part.logprobs:
                return list(part.logprobs)
    return None


def extract_arms(lps, n_arms: int = N_ARMS,
                 sentinel: str = CHOICE_SENTINEL,
                 ctx_tokens: int = 8,
                 floor_penalty: float = Z_MEM_PENALTY):
    """从 logprobs 列表里抽出 N 个候选臂的分布。

    lps : collect_logprobs() 的返回值，list[Logprob]
          每个元素有 .token / .logprob / .top_logprobs

    返回 (arm_lps, chosen, debug)；找不到返回 (None, None, {})
        arm_lps : {arm_id: logprob}，**键域恒为 1..n_arms**（缺的用地板值补）
        chosen  : 模型实际采样出的 arm 编号
        debug   : {"pos": 位置, "missing": 缺失的臂, "found": 实际找到的臂}

    判据的三处细节都是实测踩出来的：

      1. 从后往前找**第一个**范围内的数字 token。
         **不要**要求"至少 2 个臂命中"—— 那个条件会误命中 CANDIDATE k: 的索引，
         于是读到的是"模型打算写几个候选"的分布，而不是"模型选哪个"的分布。
         CHOICE 是最后一个数字 token，所以"从后往前第一个"才对。

      2. 前文 ctx_tokens 个 token 里必须出现 sentinel —— 排除 CANDIDATE 行。

      3. arm 解析一律走 parse_arm（ASCII + 无前导零）。
    """
    sent = sentinel.upper()
    for i in range(len(lps) - 1, -1, -1):
        chosen = parse_arm(lps[i].token, n_arms)
        if chosen is None:
            continue
        ctx = "".join(lps[j].token for j in range(max(0, i - ctx_tokens), i)).upper()
        if sent not in ctx:
            continue

        found = {}
        for t in lps[i].top_logprobs:
            k = parse_arm(t.token, n_arms)
            if k is not None:
                found[k] = t.logprob

        # 采样出来的 token 如果在自己的 top_logprobs 里，值一定相同；
        # 不在的话用它的 logprob 兜底（这个值是权威的）。
        if chosen not in found:
            found[chosen] = lps[i].logprob

        # 补全缺的臂：top_logprobs 里有名额被 <eos> 之类的非数字 token 占了，
        # 导致某个候选没出现。它的概率实际上极低，用比最小观测值再低一档的
        # 地板值补上，让记忆仍然有机会把它抬起来。
        lo, hi = min(found.values()), max(found.values())
        span = (hi - lo) or 1.0
        floor = lo - floor_penalty * span      # 归一化后 ≈ -1 - penalty，与 score_arms 的注入地板一致
        arm_lps = {k: found.get(k, floor) for k in range(1, n_arms + 1)}

        missing = sorted(set(range(1, n_arms + 1)) - set(found))
        return arm_lps, chosen, {"pos": i, "missing": missing, "found": sorted(found)}

    return None, None, {}
