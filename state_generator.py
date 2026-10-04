from openai import OpenAI

state_generator_prompt="""
你是一个阶段总结器，你需要总结任务进行到了哪个阶段，不需要过多强调历史信息

提供素材:
    部分的历史记录
工作方式:
    通过历史记录返回一段文本，该文本需要描述现在任务进行到了哪个阶段
输出要求：
    文本尽量的简短(<1024token)，需要包含阶段关键词如(开始，正在进行，结束等)
    例子：
        已完成对用户需求的总结，正在规划执行方案
"""


def state_generator(client:OpenAI,history:list,recent:int = 10)->str:

    resp=client.responses.create(
        model="deepseek-flash",
        instructions=state_generator_prompt,
        input=history,
        reasoning={
            "effort":"none"
        },
        stream=True,
    )
    reply=[]
    for item in resp:
        if item.type=="response.output_text.delta":
            reply.append(item.delta)

    reply="".join(reply)

    print("=======已生成state========")
    print(reply)
    print("=========================")
        
    return reply