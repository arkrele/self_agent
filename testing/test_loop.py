from task_generator import collector_llm,compactor_llm
from choice_generator import choice_generator
from decision_maker import decision_maker
from toolsCall_node import toolcall
from openai import OpenAI
from dotenv import load_dotenv
from tool_schema_class import ToolContext,Tool
from state_generator import state_generator
from evaluator import evaluator
from evaluator import evaluator,SARturple
from db_node import jitRL_DBClass,jitRL_DBParams
import os 



load_dotenv()

BETA=float(os.environ["BETA"])

def loop(toolsDict:dict[str,Tool],ctx:ToolContext):
    history:list=[]
    client=OpenAI(
        base_url=os.environ["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    collector_llm(client=client,history=history)
    task=compactor_llm(client=client,history=history)
    s_a:list[SARturple]=[]
    db_client:jitRL_DBClass=jitRL_DBClass(jitRL_DBParams(
        db_path="./jitRL_db.db",
        emb_dim=1024,
        emb_url=os.environ["EMBEDDING_URL"],
        emb_key=os.environ["EMBEDDING_KEY"],
        emb_model="BAAI/bge-large-zh-v1.5",
        col_name="testing"
    ))



    while True:
        s_a=s_a+agent_loop(task=task,history=history,toolsDict=toolsDict,client=client,ctx=ctx,db_client=db_client)
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
    sar:list[SARturple]=evaluator(client,s_a,task)
    db_client.add(s_a_r=sar,task=task)
    pass

def agent_loop(task:str,history:list,toolsDict:dict[str,Tool],client:OpenAI,ctx:ToolContext,db_client:jitRL_DBClass)->list[tuple]:

    s_a:list[SARturple]=[]
    while True:
        state=state_generator(history=history,client=client)
        actions=choice_generator(client=client,history=history,option_num=5,tools=[])
        act_lp=decision_maker(client=client,history=history,actions=actions)
        db_reward=db_client.get_A(task=task,state=state,actions=[action.model_dump_json() for action in actions])

        print("======db_query_test======")
        for i,action in enumerate(actions):
            print(f"action:{action.model_dump_json()}")
            print(f"reward:{db_reward[i]}")
        print("=========================")

        new_act_lp=[(lp[0],lp[1]+BETA*db_reward[lp[0]]) for lp in act_lp]
        best_choice=max(new_act_lp,key=lambda t :t[1])[0]
        
        act=actions[best_choice]
        history.append({
            "role":"assistant",
            "content":act.model_dump_json()
        })
        s_a.append(SARturple(
            state=state,
            action=act.model_dump_json(),
            reward=0,
        ))

        if act.text:
            print(act.text)
        if act.return_type=="no_tool":
            break
        else:
            resp=toolcall(toolsDict=toolsDict,action=act,ctx=ctx)
            history.append(
                {
                    "role":"user",
                    "content":"工具结果: "+resp
                }
            )

    return s_a





if __name__ =="__main__":
    ctx=ToolContext(
        cwd=os.getcwd(),
        readed_file=[],
    )
    toolsDict=dict()
    loop(toolsDict=toolsDict,ctx=ctx)
