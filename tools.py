import pathlib
import os
import subprocess

TOOLS = [
    {
        "type": "function",
        "name": "list_files",
        "description": "列出指定目录下的文件和子目录。当你需要了解项目结构、"
        "或者不确定某个文件是否存在时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录的相对路径，默认为当前目录",
                }
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "读取指定路径的文本文件全文。当你需要查看文件内容时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件的相对路径，例如 src/main.py",
                }
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "把内容写入指定文件（文件不存在则创建，存在则整体覆盖）。"
        "当你需要创建新文件或重写文件时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件的相对路径"},
                "content": {"type": "string", "description": "要写入的完整内容"},
            },
            "required": ["path", "content"],
        },
    },
    {   
        "type": "function",
        "name": "edit_file",
        "description": "在文件中做一次精确的字符串替换。old_str 必须在文件中恰好出现一次；"
        "如果出现多次，请提供更长的、包含上下文的唯一片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件的相对路径"},
                "old_str": {"type": "string", "description": "要被替换的原文片段，必须与文件内容逐字符一致（含缩进）"},
                "new_str": {"type": "string", "description": "替换后的新内容"},
            },
            "required": ["path", "old_str", "new_str"],
        }
    },
    {   
        "type": "function",
        "name": "run_bash",
        "description": "在当前目录执行一条 shell 命令，返回退出码和输出。"
        "用于运行代码、执行测试、使用 git 等。写完代码后应当用它验证。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"}
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "glob",
        "description": "根据通配符模式搜索文件路径，返回匹配的文件列表。"
        "适用于按文件名或扩展名查找文件，例如 '**/*.py'、'src/**/*.ts'。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 匹配模式，例如 '**/*.py'、'src/**/test_*'",
                },
                "path": {
                    "type": "string",
                    "description": "搜索的根目录，默认为当前目录",
                },
            },
            "required": ["pattern"],
        },
    },
    
    
]

read_files: set[str] = set()


def truncate_text(text: str) -> str:
    if len(text) > 1500:
        return text[:1500] + "..."
    return text
def save_path(path: str) -> pathlib.Path:
    root = pathlib.Path.cwd().resolve()
    target = (root / path).resolve()
    if target.is_relative_to(root):
        return target
    raise ValueError(f"{path} is not in {root}，超出工作目录，拒绝访问")


def list_files(path: str = ".") -> str:
    try:
        target = save_path(path)
        if not target.is_dir():
            return f"错误：{path} 不是一个目录"
        entries = [str(p.relative_to(target)) for p in target.iterdir()]
        return "\n".join(entries) if entries else "（空目录）"
    except Exception as e:
        return f"list_files 执行失败：{e}"


def read_file(path: str) -> str:
    try:
        target = save_path(path)
        if not target.is_file():
            return f"错误：{path} 不是一个文件或不存在"
        read_files.add(str(target))
        return target.read_text(encoding="utf-8")
    except Exception as e:
        return f"read_file 执行失败：{e}"


def write_file(path: str, content: str) -> str:
        target = save_path(path)
        if target.exists():
            return f"错误：{path} 文件已存在,修改已有文件请使用edit_file"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        read_files.add(str(target))
        return f"已创建文件：{path}"

def edit_file(path: str, old_str:str,new_str:str) -> str:
    target = save_path(path)
    if target not in read_files:
        return f"错误：请先使用read_file读取文件,再进行编辑"
    text = target.read_text(encoding="utf-8")
    old,new = old_str,new_str
    count = text.count(old)
    if count == 0:
        return f"错误：old_str 在文件中未找到。请重新读取文件，确认内容逐字符一致（含缩进）"
    if count > 1:
        return f"错误：old_str 在文件中出现了 {count} 次。请提供更长的、包含上下文的唯一片段"
    target.write_text(text.replace(old, new,1), encoding="utf-8")
    return f"已替换完成"

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell = True,
            capture_output=True,
            text=True,
            timeout=120,  # 没有超时的 bash 工具 = 挂死程序的按钮
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired as e:
        return f"[ERROR] 运行超时: {e}"
    output = (result.stdout + result.stderr).strip() or "无输出"
    return f"退出码：{result.returncode}\n{truncate_text(output)}"

def glob(pattern: str, path: str = ".") -> str:
    try:
        target = save_path(path)
        if not target.is_dir():
            return f"错误：{path} 不是一个目录"
        matches = sorted(target.glob(pattern))
        if not matches:
            return "未找到匹配的文件"
        results = [str(p.relative_to(pathlib.Path.cwd())) for p in matches if p.is_file()]
        if not results:
            return "未找到匹配的文件"
        return "\n".join(results[:50]) + (f"\n...（共 {len(results)} 个，仅显示前 50 个）" if len(results) > 50 else "")
    except Exception as e:
        return f"glob 执行失败：{e}"

def grep(pattern: str, include: str = "", path: str = ".") -> str:
    try:
        target = save_path(path)
        if not target.is_dir():
            return f"错误：{path} 不是一个目录"
        cmd = ["rg", "--no-heading", "--line-number", "--color", "never",
               "--max-count", "10", pattern, str(target)]
        if include:
            cmd.extend(["--glob", include])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 1:
            return "未找到匹配的内容"
        if result.returncode == 2:
            return f"grep 执行出错：{result.stderr.strip()}"
        return truncate_text(result.stdout.strip())
    except FileNotFoundError:
        return "错误：未找到 ripgrep (rg)，请先安装：winget install BurntSushi.ripgrep"
    except subprocess.TimeoutExpired:
        return "错误：grep 搜索超时"
    except Exception as e:
        return f"grep 执行失败：{e}"


    

TOOL_FUNCS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_bash": run_bash,
    "glob": glob,
    "grep": grep,
}


def run_tool(name: str, tool_input: dict) -> str:
    if name not in TOOL_FUNCS:
        raise ValueError(f"{name} is not a valid tool name")
    return TOOL_FUNCS[name](**tool_input)