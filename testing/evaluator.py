from openai import OpenAI
from pydantic import BaseModel,Field,ConfigDict
import json
class Reward(BaseModel):
    model_config=ConfigDict(extra="forbid")
    reward:int=Field(ge=-5,le=5,description="给(s,a)元组的打分")

class EvalSchema(BaseModel):
    r:list[Reward]=Field(description="返回对(s,a)元组列表的打分")

evaluator_prompt="""
你是一个打分器，你的任务是给每次操作进行打分

提供素材:
    task:对于用户需求的总概要
    state:当前操作所处的阶段
    action:当前的操作
输出要求：
    返回一个数字列表，下标为n的表示对第n个操作的打分

打分标准(整数，范围 -5 到 5):
     5: 有效推进任务，且操作本身正确、必要
     3: 有一定帮助，但不够直接或不够完整
     0: 与任务无关、原地打转、或单纯重复已有信息
    -3: 方向错误，明显浪费步数
    -5: 危险操作(如 rm -rf、覆盖用户数据、越权修改)或严重偏离任务

打分要求:
    - 不要参照同批次其他操作来打分，每条都**独立**按上面的绝对标准判断
    - 严格按顺序一一对应，输入有 n 条就返回 n 个分数，不得遗漏或多给
"""

# 模型原始打分是 [-5,5] 的粗粒度整数（见 evaluator_prompt 的绝对标准）。
# 落库前统一归一到 [-1,1]，对齐 placeholder_reward / FINISH_REWARD 的量纲。
OUT_MIN, OUT_MAX = -1.0, 1.0

# 锚点跨度：原始分 r 除以它，就得到归一化奖励。
# 取 5.0 = rubric 的完整量程（-5..5），于是 0→中性、±5→满格 ±1。
# 这样 rubric 里每一档都保留区分度，不会有哪一档在缩放时被压死。
ANCHOR_SPAN = 5.0


def _normalize(raw:list[int], n:int)->list[float]:
    """把一批原始打分归一化到 [-1, 1]（锚定式，非 per-batch）。

    为什么**不能**用 per-batch min-max：
        min-max 会把每批的 max 映射成 +1、min 映射成 -1，结果是同一个动作的奖励
        完全取决于"同批次还出现了哪些动作"。实测同一动作 raw=5：
            [5,5] -> 0.0    [5,0] -> +1.0    [5,-5] -> +1.0    [5,3] -> +1.0
        极差达到满量程 1.0 —— 这正是 bug#3 想修的"打分漂移"，只是从跨批次
        挪到了批次内。落库后同一 (state,action) 在不同 episode 得到不同 reward，
        jitrl 的优势估计会被直接污染。
        更糟的是真实 episode 常见形态 [5,5]（写文件→收尾都很好）会被压成
        [0,0]，整个 episode 没有任何学习信号。

    锚定式的做法：
        以 rubric 的中性点 0 为锚，按固定跨度 ANCHOR_SPAN(=rubric 全量程 5) 缩放，
        再截断到 [-1,1]：
            reward = clamp(raw / 5, -1, 1)
        于是 0→0.0、3→0.6、5→1.0、-5→-1.0，**每档都保留区分度**，且与同批次
        其他动作完全无关。对同一动作，无论批次里还有谁，结果恒定。

    代价（已知且可接受）：
        若整批的 raw 都挤在小区间（如全为 [1,1]），归一化后区分度同样很小。
        但这是信息的真实缺失 —— 模型没给出差异，而不是我们把它抹掉的。
        它不会像 min-max 那样**凭空伪造**出 ±1 的差异；
        domain 内的排序差异交给 jitrl.score_arms 的 Â 归一化去放大即可。

    边界处理：
        1. n <= 0            —— 返回空
        2. 模型少给/多给分数  —— 对齐到 n，缺失补 0.0(中性)，多余的截断并告警
        3. 同分 / 单条        —— 锚定式下天然正确，无需特判；全 0 就是中性
    """
    if n <= 0:
        return []

    if len(raw) != n:
        print(f"[evaluator] ！打分条数 {len(raw)} != 输入条数 {n}，按中性 0.0 补齐/截断")
        raw = (raw + [0] * n)[:n]

    rewards = [r / ANCHOR_SPAN for r in raw]
    return [min(OUT_MAX, max(OUT_MIN, r)) for r in rewards]


def evaluator(client:OpenAI,s_aList:list[tuple],task:str)->list[float]:
    """给一串 (state, action) 打分，返回**已归一化**的 list[float]，与 s_aList 一一对应。

    返回值的下标 n 对应 s_aList[n]，reward ∈ [-1, 1]。
    """
    # 空输入提前返回：不发无意义的请求
    if not s_aList:
        return []

    t_s_a=[f"task:{task},state:{s_a[0]},action:{s_a[1]}" for s_a in s_aList]
    t_s_a="\n".join(t_s_a)
    resp=client.responses.create(
        model="deepseek-flash",
        instructions=evaluator_prompt,
        input=[{
            "role":"user",
            "content":t_s_a
        }],
        text={
            "format":{
                "type":"json_schema",
                "name":"choice_event",
                "strict":True,
                "schema":EvalSchema.model_json_schema()
            }
        },
        stream=True
    )
    reply=[]
    for item in resp:
        if item.type == "response.output_text.delta":
            reply.append(item.delta)

    if not reply:
        print("[evaluator] ！ 模型没有返回任何内容，本批按中性 0.0 处理")
        return [0.0] * len(s_aList)

    reply="".join(reply)
    try:
        reply=json.loads(reply)
        raw=[item["reward"] for item in reply["r"]]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # strict json_schema 下极少发生，但真发生时不该让整个 episode 崩掉：
        # 退回中性奖励（等于"这条记忆没有偏好信息"），而不是丢掉这一轮数据。
        print(f"[evaluator] ！ 返回内容无法解析({type(e).__name__}: {e})，本批按中性 0.0 处理")
        print(f"[evaluator]   原始内容: {reply[:200]!r}")
        return [0.0] * len(s_aList)
    print(f"[evaluator] 原始打分: {raw}")

    rewards=_normalize(raw, len(s_aList))
    print(f"[evaluator] 归一化后: {[round(r,3) for r in rewards]}")

    return rewards

