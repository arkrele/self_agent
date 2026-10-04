from task_generator import collector_llm,compactor_llm
from choice_generator import choice_generator
from decision_maker import decision_maker
from toolsCall_node import toolcall
from openai import OpenAI
from dotenv import load_dotenv
from tool_schema_class import ToolContext,Tool
from state_generator import state_generator
import os 
load_dotenv()
def loop(toolsDict:dict[str,Tool],ctx:ToolContext):
    history:list=[]
    client=OpenAI(
        base_url=os.environ["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    collector_llm(client=client,history=history)
    task=compactor_llm(client=client,history=history)

    while True:
        state=state_generator(history=history,client=client)
        agent_loop(history=history,toolsDict=toolsDict,client=client,ctx=ctx)
        print("-----------------")
        try:
            user_input=input()
            if user_input =="exit":
                break
            else :
                history.append({"role":"user","content":user_input})
        except (EOFError, KeyboardInterrupt):
            break
        print("-----------------")
        pass    
        
    pass
def agent_loop(history:list,toolsDict:dict[str,Tool],client:OpenAI,ctx:ToolContext):
    while True:
        actions=choice_generator(client=client,history=history,option_num=5,tools=[])
        act_lp=decision_maker(client=client,history=history,actions=actions)
        act=actions[act_lp[0][0]]
        history.append({
            "role":"assistant",
            "content":act.model_dump_json()
        })
        if act.text:
            print(act.text)
        if act.return_type=="no_tool":
            return
        else:
            resp=toolcall(toolsDict=toolsDict,action=act,ctx=ctx)
            history.append(
                {
                    "role":"user",
                    "content":"工具结果: "+resp
                }
            )





if __name__ =="__main__":
    ctx=ToolContext(
        cwd=os.getcwd(),
        readed_file=[],
    )
    toolsDict=dict()
    loop(toolsDict=toolsDict,ctx=ctx)
