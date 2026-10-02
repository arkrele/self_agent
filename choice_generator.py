from openai import OpenAI
from pydantic import BaseModel,ConfigDict,Field
from enum import Enum
import json
import os
from dotenv import load_dotenv
class Action_Type(str,Enum):
    has_tool="has_tool"
    no_tool="no_tool"


class Choice_Action(BaseModel):
    model_config=ConfigDict(extra="forbid")

    return_type:Action_Type=Field(description="该Action的类型")
    text:str=Field("",description="输出给用户的文本")
    tool_name:str=Field("",description="即将使用的工具")    
    tool_arg:str=Field("",description="工具使用的参数")
    index:int=Field(description="该Action输出的顺序序号")

class ChoiceSchema(BaseModel):
    actions:list[Choice_Action]    

choice_generator_prompt="""
你现在是一个子agent,为主agent提供输出选项,并且需要严格按照json格式输出
你现在可用的工具有
    - read : 参数为path 路径
返回选项个数要求:
"""

def choice_generator(client:OpenAI,history:list,option_num:int)->list[Choice_Action]:
    resp=client.responses.create(
        model="deepseek-flash",
        instructions=choice_generator_prompt+f"{option_num}",
        input=history,
        text={
            "format":{
                "type":"json_schema",
                "name":"choice_event",
                "strict":True,
                "schema":ChoiceSchema.model_json_schema()
            }
        },
        stream=True
    )
    actions=[]

    for event in resp:
        if event.type == "response.output_text.delta":
            actions.append(event.delta)

    actions="".join(actions)
    actions=json.loads(actions)
    return actions

if __name__=="__main__":
    load_dotenv()
    client=OpenAI(
        base_url=os.environ["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"]
    )    
    user_input=input()
    option_num=input("请输入返回个数")
    history=[]
    history.append({"role":"user","content":user_input})
    actions=choice_generator(client,history=history,option_num=option_num)
    print(actions)
