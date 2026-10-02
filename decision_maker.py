from openai import OpenAI
from choice_generator import Choice_Action,choice_generator
from dotenv import load_dotenv
import os 
import re

decision_maker_prompt="""
你现在是一个决定者的角色，选项提供者已经为你提供了多个可执行选项，你需要根据上下文判断哪个是最优解

输出要求：
    - 只要一个阿拉伯数字（禁止输出其他字符）,表示最优解的index
"""
def decision_maker(client:OpenAI,history:list,actions:list[Choice_Action]):
    actions_str=[action.model_dump_json() for action in actions]
    actions_str="\n".join(actions_str)
    new_history=history+[{"role":"user","content":actions_str}]

    resp=client.responses.create(
        model="deepseek-flash",
        instructions=decision_maker_prompt,
        input=new_history,
        top_logprobs=len(actions) if len(actions)<=20 else 20
    )

    choice_logprob:list[tuple]=[]
    for item in resp.output:
        if hasattr(item, "content"):
            for content in item.content:
                if hasattr(content, "logprobs") and content.logprobs:
                    for entry in content.logprobs:
                        if entry.top_logprobs:
                            print(entry.top_logprobs,"\n")
                            for alt in entry.top_logprobs:
                                match_num=re.search(r"-?\d+",alt.token)
                                if match_num:
                                    choice_logprob.append((int(match_num.group()),float(alt.logprob)))
    return choice_logprob  

    
if __name__=="__main__":
    load_dotenv()
    client=OpenAI(
        base_url=os.environ["base_url"],
        api_key=os.environ["DEEPSEEK_API_KEY"]
    )    
    user_input="我想写一本百合小说，请给我一个方案"
    option_num=5
    history=[]
    history.append({"role":"user","content":user_input})
    actions=choice_generator(client,history=history,option_num=option_num)
   
    print(actions) 
    choice_logprob=decision_maker(client=client,history=history,actions=actions)
    print(choice_logprob)