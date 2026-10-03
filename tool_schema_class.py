from pydantic import BaseModel,Field,ConfigDict
from typing import Callable
from dataclasses import dataclass,field
from pathlib import Path
import subprocess
@dataclass
class ToolContext:
    cwd:Path
    readed_file:list[str]=field(default_factory=list)


@dataclass
class Tool:
    name:str
    description:str    
    params:type[BaseModel]
    execute:Callable[[BaseModel],str]

    def declaration(self)->dict:
        schema=self.params.model_json_schema()
        schema.pop("title",None)
        return {
            "type":"function",
            "name":self.name,
            "description":self.description,
            "parameters":schema,
        }
#测试样例，无实际用途，进行开发时不要引用
class BashParams(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout: float | None = Field(None, description="Timeout in seconds (optional)")

def _bash(p:BashParams,ctx:ToolContext):
    pass

bash_tool=Tool(
    name="bash",
    description="Execute a bash command in the current working directory. Returns the exit code plus stdout and stderr as text.",
    params=BashParams,
    execute=_bash
)