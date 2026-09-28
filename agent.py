import os
import pathlib
from openai import OpenAI
import json
from tools import TOOLS,run_tool
from dotenv import load_dotenv
import platform

load_dotenv()

model =  "deepseek-flash"

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


def compact(cilent:OpenAI,history_message:list) -> list:
    "压缩上下文"

    print("[系统]: 压缩上下文")

    response = cilent.responses.create(
        model = model,
        instructions = system_prompt,
        input = history_message +[{"role": "user", "content": SUMMARIZE_PROMPT}], #type: ignore
        stream = False,
    )


    summary =  "".join(content.text for content in response.output if content.type == "message"
                       for content in content.content
                       if content.type == "text"
                       )

    return [
        {
            "role": "user",
            "content": f"[以下是之前对话的摘要，请基于它继续工作]\n\n{summary}",
        }
    ]




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


def save_path(path: str)->pathlib.Path:
    """把模型给的路径限制在当前工作目录内，防止越界访问（如 ../../etc/passwd）。
    """
    root = pathlib.Path.cwd().resolve()
    target = (root/path).resolve()
    if target.is_relative_to(root):
        return target
    raise ValueError(f"{path} is not in {root},超出工作目录,拒绝访问")



def agent_run(cilent: OpenAI,prompt: str,system_prompt: str)->None:
    """处理一轮用户输入：循环调用模型和工具，直到模型给出最终回答。"""
    conversation_history = [
        {"role": "user", "content": prompt},
    ]
    while True:
        stream = cilent.responses.create(
            model = model,
            instructions= system_prompt,
            input = conversation_history, #type: ignore
            tools = TOOLS, #type: ignore
            stream = True,
        )


        output_items = []
        func_calls = []

        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)

            elif event.type == "response.function_call_arguments.delta":
                pass

            elif event.type == "response.output_item.completed":
                item = event.item
                if item.type == "message":
                    print()
                elif item.type == "function_call":
                    function_args = json.loads(item.arguments)
                    print(f"\n[工具调用]: {item.name} ({function_args} )")
                    func_calls.append({
                        "call_id": item.call_id,
                        "name": item.name,
                        "args": function_args,
                    })

            elif event.type == "response.completed":
                for item in event.response.output:
                    output_items.append(item.model_dump())

        conversation_history.extend(output_items)

        for fc in func_calls:
            result = run_tool(fc["name"], fc["args"])
            print(f"[工具返回]: {result}")
            conversation_history.append({
                "type": "function_call_output",
                "call_id": fc["call_id"],
                "output": result,
            })

        if not func_calls:
            break

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("请设置环境变量 OPENAI_API_KEY")
        raise SystemExit

    cilent = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    "输入exit退出程序"
    while True:
        try:
            user_input = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() == "exit":
            break
        prompt = build_system_prompt()
        agent_run(cilent, user_input,prompt)
        print()

if __name__ == "__main__":
    main()
