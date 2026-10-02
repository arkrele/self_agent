"""JitRL 的 agent 主循环：把 jitrl.py / db_operation.py / tools.py 串起来。

    agent.py   原生 function calling 的基线（无记忆），保留作为对照
    loop.py    JitRL：文本协议 + N 个候选臂 + 记忆修正

本地跑（会把记忆写进 jitrl_memory.db）：
    python loop.py --task "查看当前目录下有哪些文件" --episodes 3
"""
import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

import jitrl
from db_operation import jitRL_DBClass, jitRL_DBParams
from tools import TOOLS, run_tool

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------- 配置 ----------------
DB_PATH          = "jitrl_memory.db"
COL_NAME         = "jitrl"
EMB_DIM          = 1024          # 必须与 local_embed 的输出维度、schema 一致
GAMMA            = 0.95          # return-to-go 的折扣（写入时固定）
BETA             = jitrl.BETA
RETRIEVAL_K      = 50            # 召回上限
# 余弦阈值。**0.9 是配 local_embed 的**：state 本来就设计成逐字符重复，
# 相同串 = 1.000，而"同一任务下不同活动"能到 0.5~0.8 —— 那是不该召回的噪声。
# ⚠️ 换成真实语义 embedding（bge-m3）后这个值必须重新标定：
#    语义向量里"同义但措辞不同"会低于 1.0，需要放低到 0.5~0.7。
#    判断依据仍然是 query_neighbors 的命中数与其中不同 action 的个数。
RETRIEVAL_RADIUS = 0.9
MAX_STEPS        = 15
FINISH_REWARD    = 1.0

INSTRUCTIONS = """你是一个运行在用户终端里的 Coding Agent。

<工作方式>
- 动手前先搞清楚现状：先列目录 / 搜文件，再看内容，不要凭空猜测
- 修改已有文件前必须先读取
- 只做用户要求的事：不顺手重构、不添加没被要求的功能
</工作方式>

<输出要求>
严格按提示词里给定的格式输出候选动作和选择，不要有任何多余内容。
</输出要求>"""


# ============ 胶水：把 tools / 历史 / 检索结果转成 jitrl 要的形状 ============

def render_tools(tools=TOOLS) -> str:
    """工具列表 → 提示词里的文本。jinja 不用，保持无依赖。"""
    out = []
    for t in tools:
        props = t["parameters"].get("properties", {})
        req = set(t["parameters"].get("required", []))
        ps = ", ".join(f"{k}{'' if k in req else '?'}" for k in props)
        out.append(f"- {t['name']}({ps}) — {t['description']}")
    out.append("- finish() — 任务已完成时使用，结束本轮")
    return "\n".join(out)


def render_records(records, limit: int = 6, max_result_chars: int = 400) -> str:
    """最近若干步的操作记录 → 提示词文本。只渲染最近 limit 条，
    否则上下文会随步数无界增长（agent.py 里靠 compact() 解决，这里靠截断）。"""
    if not records:
        return "(还没有任何操作)"
    start = max(1, len(records) - limit + 1)
    lines = []
    for i, (tool, args, result) in enumerate(records[-limit:], start=start):
        lines.append(f"步骤{i}: {tool}({json.dumps(args, ensure_ascii=False)})")
        lines.append(f"  结果: {' '.join(str(result).split())[:max_result_chars]}")
    return "\n".join(lines)


def local_embed(text: str) -> list[float]:
    """【占位】token 哈希 embedding：只有词面重合，没有语义。

    ⚠️ 之所以能顶用：summarize_state 产出的 state 是高度结构化的短串
       （"task=xxx | glob:ok → run_bash:err"），共享 token 多，词面相似度
       与语义相似度在这类串上基本等价。

    ⚠️ 真实使用必须换成 bge-m3（DeepSeek 没有 embedding 接口）。
       换的时候：
         1. build_db 里填对 emb_url / emb_key / emb_model / emb_dim
         2. 去掉 `db.get_embVector = local_embed` 那行覆盖
         3. **删掉旧库**（jitrl_memory.db）—— 新旧向量不在同一空间，
            混在一起检索会得到垃圾结果，而且不会报错

    注意 hash 必须用 md5 而不是内置 hash()：后者对 str 是按进程随机的
    （PYTHONHASHSEED），会导致写入和查询算出不同的向量。
    """
    v = [0.0] * EMB_DIM
    # 英文/数字按词切，中文按字切，其余按符号切
    for tok in re.findall(r"[A-Za-z0-9_]+|[一-鿿]|[^\sA-Za-z0-9_]", text.lower()):
        # 跳过纯标点（':' '→' '|' 等）：它们被所有 state 共享，
        # 不携带信息却会把整体相似度抬高 —— 实测会让
        # "glob:ok → run_bash:err" 和 "read_file:ok → edit_file:ok" 的余弦达到 0.70。
        if not tok.isalnum():
            continue
        h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "big")
        v[h % EMB_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def aggregate_q(neighbors) -> dict:
    """[Neighbor, ...] → {action: [g, ...]}，直接喂给 score_arms。"""
    q = defaultdict(list)
    for nb in neighbors:
        q[nb.action].append(nb.g)
    return dict(q)


def placeholder_reward(client:OpenAI,tool: str, args: dict, result: str, records) -> float:
    """【占位奖励】⚠️ 下一步会被 episode 末的 LLM 评估器替换。

    它只知道"这一步有没有报错"，**不知道"有没有推进任务"** —— coding agent
    没有环境 reward。所以用它只能验证机制（记忆能否翻转决策），学不到策略。
    """
    instruction="""
    你是一个打分器,你需要根据tool,args,result三个参数为这次工具调用打分,
    三个参数的含义如下
        tool:调用的工具
        args:传入的参数
        result:工具返回的结果
    输出约束：
        只输出一个数字，范围为[-1,1]的整数，表示这次行为的分数
    """
    input=f"""
        tools:{tool},
        args:{args},
        result:{result}
    """
    resp=client.responses.create(
        model="deepseek-flash",
        instructions=instruction,
        input=input
    )
    print("-------------")
    print("操作打分:",resp.output_text)
    print("-------------")        
    r=int(resp.output_text)
    return r


# ============ 主循环 ============

def build_db(db_path: str = DB_PATH) -> jitRL_DBClass:
    db = jitRL_DBClass(jitRL_DBParams(
        db_path=db_path, emb_url="http://unused", emb_key="unused",
        col_name=COL_NAME, emb_model="local-hash", emb_dim=EMB_DIM,
    ))
    db.get_embVector = local_embed     # ← 接真实 embedding 服务时删掉这行
    return db


def run_episode(client, db, task: str, episode_id: int, verbose: bool = True,
                max_steps: int = MAX_STEPS):
    """跑一个 episode（同一任务的一次完整尝试），结束后把 (s, a, r) 序列写进记忆。

    每步的决策链路：
        summarize_state → query_neighbors → aggregate_q
        → build_prompt → step_completion → parse_candidates / extract_arms
        → score_arms → argmax → parse_sig → run_tool
    """
    records: list[tuple] = []
    trace: list[tuple] = []
    # 任务指纹：写入时存进 task 字段，检索时按它硬过滤。
    # 不同任务的 G 不可比 —— 不隔离的话 V̂ 基线会被别的任务污染。
    task_fp = jitrl.task_fingerprint(task)

    for step in range(1, max_steps + 1):
        # state 里不含任务 —— 任务由 query_neighbors 的 task 参数硬过滤。
        # 完整格式见 jitrl.summarize_state 的 docstring（那里记了为什么）。
        state = jitrl.summarize_state(records)
        observation = records[-1][2] if records else "(开始)"

        prompt = jitrl.build_prompt(task, render_tools(),
                                    render_records(records), observation, jitrl.N_ARMS)
        try:
            resp = jitrl.step_completion(client, INSTRUCTIONS, prompt)
        except jitrl.TruncatedResponse as e:
            print(f"[!] {e}")
            break

        cands = jitrl.parse_candidates(resp.output_text, jitrl.N_ARMS)
        if cands is None:
            print("[!] 候选解析失败，终止本轮")
            print("    " + resp.output_text[:300].replace("\n", " "))
            break

        lps = jitrl.collect_logprobs(resp)
        arm_lps, chosen, _ = jitrl.extract_arms(lps, jitrl.N_ARMS) if lps else (None, None, {})
        if arm_lps is None:
            print("[!] logprob 提取失败，终止本轮")
            break

        texts = {k: jitrl.action_sig(*cands[k]) for k in cands}
        q = aggregate_q(db.query_neighbors(state, limit=RETRIEVAL_K,
                                           radius=RETRIEVAL_RADIUS, task=task_fp))
        scores, sdbg = jitrl.score_arms(arm_lps, texts, q, beta=BETA)
        # ⚠️ 用 sdbg["texts"] 而不是上面的 texts：记忆独有的动作会被注入成新臂
        # （id > N_ARMS），原始 texts 里没有它们，直接查会 KeyError。
        texts = sdbg["texts"]
        best = max(scores, key=lambda k: scores[k])
        sig = texts[best]

        parsed = jitrl.parse_sig(sig)
        if parsed is None:
            print(f"[!] 签名无法还原: {sig}")
            break
        tool, args = parsed

        if verbose:
            n_hit = sum(len(v) for v in q.values())
            print(f"\n[第{step}步] state = {state}")
            print(f"  记忆: 召回 {n_hit} 条 / {len(q)} 个不同动作   V̂={sdbg['V']:.2f}")
            print(f"  模型自选 = {chosen}  {texts[chosen]}")
            print(f"  → 最终   = {best}  {sig}"
                  + ("   <<< 记忆翻转" if best != chosen else ""))

        if tool == "finish":
            trace.append((state, sig, FINISH_REWARD))
            break

        try:
            result = run_tool(tool, args)
        except Exception as e:
            result = f"[ERROR] 工具执行异常：{e}"
        if verbose:
            print(f"  结果: {' '.join(str(result).split())[:200]}")

        trace.append((state, sig, placeholder_reward(client,tool, args, result, records)))
        records.append((tool, args, result))
    else:
        print(f"[!] 达到最大步数 {max_steps}")

    if trace:
        db.add_episode(episode_id, trace, task_fp, gamma=GAMMA)
    print(f"\n[本轮结束] {len(trace)} 步，累计奖励 {sum(r for _, _, r in trace):.2f}，"
          f"记忆库共 {db.count()} 条")
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="任务描述")
    ap.add_argument("--episodes", type=int, default=1, help="重复跑几轮（JitRL 靠跨轮积累）")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    a = ap.parse_args()

    key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("请设置环境变量 API_KEY"); raise SystemExit
    client = OpenAI(api_key=key,
                    base_url=os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
                    max_retries=0, timeout=120)

    db = build_db(a.db)
    print(f"记忆库: {a.db}  已有 {db.count()} 条\n")

    for i in range(a.episodes):
        print(f"{'='*70}\nEpisode {a.start_id + i + 1}   任务: {a.task}\n{'='*70}")
        run_episode(client, db, a.task, a.start_id + i + 1,
                    verbose=not a.quiet, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
