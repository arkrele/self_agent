from tool_schema_class import Tool,ToolContext
from choice_generator import Choice_Action
import json

def toolcall(toolsDict:dict[str,Tool],action:Choice_Action,ctx:ToolContext)->str:
    if action.return_type=="no_tool":
        return "无工具调用"

    try:
        tool=toolsDict[action.tool_name]
        tool_arg=tool.params.model_validate(json.loads(action.tool_arg))
        resp=tool.execute(tool_arg,ctx)
        return resp
    except Exception as e :
        return f"{type(e).__name__}:{e}"
