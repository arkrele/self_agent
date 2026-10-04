from openai import OpenAI
from pydantic import BaseModel,ConfigDict,ValidationError,Field
from colorama import Fore
import os
from dotenv import load_dotenv
import json
class ReadinessJudgment(BaseModel):
    ready_to_conclude:bool=Field(description="一个布尔值，表示是否收集到祖国的信息")
    missing_info:str=Field(description="给用户的对话信息，进一步提问")
    confidence:float=Field(description="表示对于成功生成用户需求的置信度(范围为[0,1],当confidence>0.8时将结束询问进入下一个阶段)")
    model_config=ConfigDict(extra="forbid")    

collector_prompt="""
你是一个运行在用户终端的子agent
你的工作目的是通过问答明确用户的需求

工作方式：
    使用提问的方式，明确用户的需求。
    可以通过提供选项的方式，让用户选择，进一步明确用户的需求
    最后生成一个task作为该需求的总结
输出方式：
    强制使用json格式输出:
        ready_to_conclude:表示已经完成问答环节(当置1时结束对话)
        confidence: 表示对于成功生成用户需求的置信度(范围为[0,1],当confidence>0.8时将结束询问进入下一个阶段)
        missing_info:下一个问题的描述(必要的时候可以做成选择题供用户选择)
安全边界：
    - 通过问答的方式，逐步明确用户的真实需求
    - 禁止使用工具

"""

compactor_prompt="""
你是一个需求总结器
你的工作是根据历史记录提取并总结用户的需求

工作方式：
    根据上下文进行总结,输出一个小于1024字符的字符串
输出限制：
    - 仅输出对历史记录的总结，不做任何工作
"""

def compactor_llm(client:OpenAI,history:list)->str:
    resp=client.responses.create(
        model="deepseek-flash",
        instructions=compactor_prompt,
        input=history,
    )
#    task=[]
#    for item in resp.output:
#        if item.type=="message":
#            for text in item.content:
#                if text.type=="output_text":
#                    task.append(text.text)
    task=resp.output_text
    print("=====已完成总结工作，用户需求如下=====")
    print(task)
    history.append({
        "role":"user",
        "content":"用户需求总结:"+task
    })
    return task



def collector_llm(client:OpenAI,history:list,limit:int=500):
    """
        history是历史记录
        limit是问答上限
    """
    index=0
    wrong_return=0
    while True:
        
        if index>=limit:
            print(Fore.RED+"已到问答上限，将进入下一阶段")    
            history.append(
                {
                    "role":"user",
                    "content":"已到问答上限，将进入下一阶段"
                }
            )
            break
        index=index+1
        print("---------------------")

        try:
            user_input=input()
            if user_input=="exit":
                break
            history.append(
                {
                    "role":"user",
                    "content":user_input,
                }
            )
        except (EOFError, KeyboardInterrupt):
                break
        if not user_input or user_input.lower() == "exit":
            print("用户已终止询问环节,即将进行task总结")
            history.append(
                {
                    "role":"user",
                    "content":"用户已终止询问环节"
                }
            )
            break
        print("---------------------")
        resp=client.responses.create(
            model="deepseek-flash",
            instructions=collector_prompt,
            input=history,
            text={
                "format":{
                    "type":"json_schema",
                    "name":"choice_event",
                    "strict":True,
                    "schema":ReadinessJudgment.model_json_schema()
                }
            },
            stream=True
        )
        reply=[]
        for item in resp:
            if item.type=="response.output_text.delta":
                reply.append(item.delta)
        reply="".join(reply)
        history.append({"role":"assistant","content":reply})
        reply=ReadinessJudgment.model_validate(json.loads(reply))
        if reply.ready_to_conclude==1 or reply.confidence>0.8:
            print("=====已收集到足够信息，即将进入下一阶段======") 
            break
        else :
            print(reply.missing_info)

if __name__=="__main__":

    load_dotenv()
    history=[]
    client=OpenAI(
        base_url=os.environ["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"]
    ) 

    collector_llm(client,history,500)
    compactor_llm(client,history)
