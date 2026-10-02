import io
import json
import os
import pathlib
import platform
import sys
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ToolParam

from tools import TOOLS, run_tool

load_dotenv()

model = "deepseek-flash"

# 历史的字符预算，超过就触发压缩。粗估即可（中英混排约 1 token ≈ 2~3 字符）。
HISTORY_CHAR_BUDGET = 600000

system_prompt = """你是一个运行在用户终端里的 Coding Agent。

<环境信息>
工作目录：{cwd}
操作系统：{os_name}
是否 git 仓库：{is_git}
目录概览：
{tree}
</环境信息>

<工作方式>
- 动手前先搞清楚现状：用 glob / grep 定位代码，用 read_file 确认内容，不要凭空猜测
- 修改已有文件用 edit_file 精确替换；编辑前必须先读取
- 写完代码用 run_bash 运行或测试，验证之后才能说"完成"
- 只做用户要求的事：不顺手重构、不添加没被要求的功能、不写多余的防御性代码
- 遇到模糊的需求，简单的歧义自己做合理决定并说明，重大分歧再向用户提问
</工作方式>

<沟通规范>
- 用简体中文回答
- 先给结论，再给必要的细节；简单问题直接一句话回答
- 汇报时只说你验证过的事实：测试没跑就说没跑，失败了就贴出失败信息
</沟通规范>

<安全边界>
- 禁止执行破坏性命令（rm -rf、强制推送、修改系统配置等）
- 不读取、不外传 .env 等可能含密钥的文件内容
</安全边界>"""

SUMMARIZE_PROMPT = """请把以上全部对话压缩成一份摘要，供你在后续对话中恢复上下文使用。必须保留：
1. 用户的原始目标和所有明确要求
2. 已经完成了什么、验证结果如何
3. 关键发现：重要文件路径、代码结构、踩过的坑
4. 未完成的事项和下一步计划
只输出摘要本身，不要任何开场白。"""


def build_client() -> OpenAI:
    """从 .env 构造 client。

    ⚠️ base_url 必须显式传。不传的话 SDK 会打到 api.openai.com，而这里配的
    是 DeepSeek 的 key —— 表现是 ConnectTimeout（连都连不上，不是 401），
    排查时非常容易误判成网络问题。

    key 名优先读 API_KEY/BASE_URL（.env.example 里的写法），
    同时兼容旧的 OPENAI_API_KEY/OPENAI_BASE_URL。
    """
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("请设置环境变量 API_KEY（见 .env.example）")
    if not base_url:
        print("[warn] 没读到 BASE_URL，将打到 api.openai.com —— 多半会连接超时")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=2)


def history_size(history: list) -> int:
    """历史的粗略规模（字符数）。只是用来决定要不要压缩，不需要精确。"""
    return sum(len(str(m)) for m in history)


def compact(client: OpenAI, history: list, system_prompt: str) -> list:
    """把历史压成一份摘要，返回压缩后的新历史（只含一条 user 消息）。"""
    print("[系统]: 压缩上下文")

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=history + [{"role": "user", "content": SUMMARIZE_PROMPT}],
        max_output_tokens=80000,
        stream=False,
    )

    parts = []
    for item in response.output:
        if item.type != "message":
            continue
        for part in item.content:
            if part.type == "text":
                parts.append(part.text)
    summary = "".join(parts)
    if not summary.strip():
        raise RuntimeError("摘要为空")

    return [
        {
            "role": "user",
            "content": f"[以下是之前对话的摘要，请基于它继续工作]\n\n{summary}",
        }
    ]


def maybe_compact(client: OpenAI, history: list, system_prompt: str) -> None:
    """历史超预算就压缩。

    只在 while 循环顶部调用 —— 那里每个 function_call 都已经配好了对应的
    function_call_output，压缩掉整段历史是结构安全的（不会留下悬空的 tool 调用）。
    中途（工具输出还没 append 时）压缩会把 function_call 和它的输出拆散，
    下次请求会直接 400。
    """
    if history_size(history) < HISTORY_CHAR_BUDGET:
        return
    try:
        history[:] = compact(client, history, system_prompt)   # 原地替换
    except Exception as e:
        # 压缩失败就带着原历史继续跑，总比把上下文清空好
        print(f"[系统]: 压缩失败，保持原历史继续：{e}")


def build_system_prompt() -> str:
    """运行环境的真实信息注入"""
    cwd = os.getcwd()
    is_git = pathlib.Path(".git").is_dir()
    try:
        entries = sorted(pathlib.Path(".").iterdir())[:30]
        tree = "\n".join(
            ("  " + p.name + ("/" if p.is_dir() else "")) for p in entries
        )
    except OSError:
        tree = "  （无法读取）"
    return system_prompt.format(
        cwd=cwd,
        os_name=f"{platform.system()} {platform.release()}",
        is_git="是" if is_git else "否",
        tree=tree,
    )


def save_path(path: str) -> pathlib.Path:
    """把模型给的路径限制在当前工作目录内，防止越界访问（如 ../../etc/passwd）。
    """
    root = pathlib.Path.cwd().resolve()
    target = (root / path).resolve()
    if target.is_relative_to(root):
        return target
    raise ValueError(f"{path} is not in {root},超出工作目录,拒绝访问")


def agent_run(client: OpenAI, user_input: str, history: list,system_prompt: str) -> None:
    """处理一轮用户输入：循环调用模型和工具，直到模型给出最终回答。

    history 由调用方持有并**跨轮次复用**，这里只往里追加、不新建 ——
    每次进来都新建一个 list 的话，上一轮说过什么就全丢了。
    """
    history.append({"role": "user", "content": user_input})

    while True:
        maybe_compact(client, history, system_prompt)

        stream = client.responses.create(
            model = model,
            instructions=system_prompt,
            input=history,
            # TOOLS 是普通 dict 字面量（tools.py 不依赖 openai），而 SDK 的
            # FunctionToolParam 把 strict 标成了 Required —— 我们不打算发这个字段
            # （各 provider 对它的默认值和严格校验要求不一致），所以类型对不上。
            # 运行时的形状是合法的，用 cast 说明这一点。
            tools=cast(list[ToolParam], TOOLS),
            stream=True,
        )

        output_items = []      # 本轮模型产出的 item（reasoning / message / function_call）
        func_calls = []
        failure = None

        for event in stream:
            etype = event.type

            if etype == "response.output_text.delta":
                print(event.delta, end="", flush=True)

            elif etype == "response.output_item.done":
                # 注意：这个事件上**没有** .response 属性（只有 item / output_index /
                # sequence_number），想拿完整 output 要用 response.completed 的 .response。
                item = event.item
                output_items.append(item.model_dump())
                if item.type == "message":
                    print()
                elif item.type == "function_call":
                    # arguments 是流式拼出来的字符串，空串或半截 JSON 都可能出现，
                    # 直接 json.loads 会抛异常打断整轮对话
                    try:
                        args = json.loads(item.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": item.arguments}
                    print(f"\n[工具调用]: {item.name} ({args})")
                    func_calls.append({
                        "call_id": item.call_id,
                        "name": item.name,
                        "args": args,
                    })

            elif etype == "response.failed":
                failure = getattr(event.response, "error", None)
            elif etype == "response.incomplete":
                failure = getattr(event.response, "incomplete_details", None)
            elif etype == "error":
                failure = getattr(event, "message", None)

        if failure is not None:
            # 响应残缺时**不写回历史**：半截的 function_call 没有配对的输出，
            # 留在历史里会让后续每一次请求都 400。
            print(f"\n[!] 本轮响应失败或被截断，已丢弃：{failure}")
            break

        # 顺序不能反：必须是「模型这一轮的全部 item」在前，「工具输出」在后。
        # 另外 output_items 里同时包含 message 和 function_call，
        # 模型在调工具前说的那句话也是历史的一部分，不能丢。
        history.extend(output_items)

        if not func_calls:
            break

        for fc in func_calls:
            try:
                result = run_tool(fc["name"], fc["args"])
            except Exception as e:
                # run_tool 对未知工具名抛 ValueError、参数不匹配抛 TypeError。
                # 不能让它冒出去：异常中断的话 function_call 就没有配对的输出，
                # 历史结构被破坏，下一次请求直接 400。
                result = f"[ERROR] 工具执行异常：{e}"
            print(f"[工具返回]: {result}")
            history.append({
                "type": "function_call_output",
                "call_id": fc["call_id"],
                "output": result,
            })

    plan_agent(history=history,client=client,system_prompt=system_prompt)
def plan_agent(history:list,client:OpenAI,system_prompt)->None:
    text_his:list[str]=[]
    for his in history:
        print(his)
        if his.get("role")=="user":
            text_his.append("role:user\n"+his.get("content")+"\n")
        if his.get("type")=="message":
            for item in his.get("content") or None:
                if item.get("type")=="output_text":
                    text_his.append("role:ai\n"+item.get("text")+"\n")

            



def main():
    # Windows 控制台默认 GBK，模型输出里一旦出现 GBK 编不出的字符
    # （✓ ← 之类的），print 会抛 UnicodeEncodeError 把 agent 直接打断。
    # 只改 errors 不改 encoding：中文仍然正常显示，编不出的字符退化成 '?'。
    # 用 isinstance 判断而不是直接调：sys.stdout 的标注类型是 TextIO，
    # 上面没有 reconfigure（那是 TextIOWrapper 的方法），直接调 Pylance 会报错。
    if isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout.reconfigure(errors="replace")
        except OSError:
            pass

    client = build_client()

    history: list = []          # 跨轮次复用 —— 上下文记忆就靠它

    "输入exit退出程序"
    while True:
        try:
            user_input = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() == "exit":
            break
        agent_run(client, user_input, history, build_system_prompt())
        print()


if __name__ == "__main__":
    main()
