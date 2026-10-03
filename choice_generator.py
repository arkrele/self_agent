from openai import OpenAI
from pydantic import BaseModel,ConfigDict,Field
from enum import Enum
import json
import os
from dotenv import load_dotenv
from tool_schema_class import bash_tool
from tool_schema_class import BashParams
class Action_Type(str,Enum):
    has_tool="has_tool"
    no_tool="no_tool"

#这里以后可以改成一次性调用好几个工具，只需要添加一个tool_call类。
# 然后Choice_Action中改成列表就好了,当然要记得改提示词哦

class Choice_Action(BaseModel):
    model_config=ConfigDict(extra="forbid")

    return_type:Action_Type=Field(description="该Action的类型")
    text:str=Field("",description="输出给用户的文本")
    tool_name:str=Field("",description="即将使用的工具")    
    tool_arg:str=Field("",description="工具使用的参数,以json格式返回，若格式错误会返回报错")
    index:int=Field(description="该Action输出的顺序序号")

class ChoiceSchema(BaseModel):
    actions:list[Choice_Action]    

choice_generator_prompt="""
你现在是一个子agent,为主agent提供输出选项,并且需要严格按照json格式输出，输出中不要体现你在为主agent提供方案，更不要出现方案的字眼
工具使用要求：
    通过文本直接传回调用的工具和工具参数，禁止通过funcion_call传回,且只能调用一个工具
返回选项个数要求:
"""

def choice_generator(client:OpenAI,history:list,option_num:int,tools:list)->list[Choice_Action]:
    resp=client.responses.create(
        model="deepseek-flash",
        instructions=choice_generator_prompt+f"{option_num}",
        input=history,
        tools=tools,
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

   
    if not actions:
        return "there is no more options"
    actions="".join(actions)
    actions=json.loads(actions)
    actions=[Choice_Action.model_validate(action) for action in actions["actions"]]
    return actions

if __name__=="__main__":
    load_dotenv()
    client=OpenAI(
        base_url=os.environ["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"]
    )    
    user_input="在test.txt中写一个 hello world.注意要传回timeout参数"
    option_num=5
    history=[]
    history.append({"role":"user","content":user_input})
    actions=choice_generator(client,history=history,option_num=option_num,tools=[bash_tool.declaration()])
    for action in actions:
        print("-----------------")
        print(action.return_type)
        print(action.text)
        print(action.tool_name)
        if action.tool_arg:
            arg=json.loads(action.tool_arg)
            arg_p=BashParams.model_validate(arg)
            print(arg_p)
        
