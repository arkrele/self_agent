from pymilvus import MilvusClient,DataType
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

load_dotenv()

MAX_STATE_LEN = 1024   # 与 getSchema() 里 state 的 VARCHAR max_length 保持一致

# 检索时要取回的标量字段。schema 改了字段名，这里必须同步改。
# 显式标注成 str（而不是让 Pylance 推断出 Literal[...]）：
# list 是不变的，list[Literal['state']] 并不是 list[str] 的子类型，
# 直接传给 Milvus 的 output_fields: List[str] 会报 reportArgumentType。
_ENTITY_FIELDS: tuple[str, ...] = ("state","action","reward","g","episode_id","step_index","task")

# 检索默认要的字段（给 Milvus 的 output_fields）
_OUTPUT_FIELDS: list[str] = list(_ENTITY_FIELDS)


def _hit_get(hit,name,default=None):
    """兼容不同 pymilvus 版本的 hit 取值方式：
    MilvusClient.search 返回 dict，老式 Collection.search 返回对象。"""
    if isinstance(hit,dict):
        return hit.get(name,default)
    return getattr(hit,name,default)


class RecordModel(BaseModel):
    state:str
    state_vec:Optional[list[float]] = None
    action:str
    action_vec:Optional[list[float]]=None
    reward:float
    g:float
    episode_id:int
    step_index:int
    done:bool
    task:str = ""        #对用户任务的描述，是一个不超过1024字符的字符串  
    task_vec:Optional[list[float]]=None

class jitRL_DBParams(BaseModel):
    db_path:str
    emb_url:str
    emb_key:str
    col_name:str
    emb_model:str = "BAAI/bge-m3"
    emb_dim:int = 1024

class jitRL_DBClass(object):
    def __init__(self,p:jitRL_DBParams):
        self.p=p
        db_path=Path(p.db_path)
        if not db_path.parent.is_dir():
            db_path.parent.mkdir(parents=True,exist_ok=True)
        self._db_client=MilvusClient(str(db_path))
        self.col_name=p.col_name
        self._emb_client=OpenAI(
            base_url=p.emb_url,
            api_key=p.emb_key
        )
        self.emb_cache: dict[str,list[float]] = {}

        if not self._db_client.has_collection(collection_name=self.col_name):
            self._db_client.create_collection(
                collection_name=self.col_name,
                schema=self.getSchema(),
                index_params=self.getIndexParam()
            )
        else:
            # Milvus 的 schema 建好之后不能改字段。集合已经存在时，
            # getSchema() 的改动**不会生效**，接着 insert 会报字段缺失。
            print(f"[warn] 集合 {self.col_name} 已存在，本次 schema 改动不会生效。"
                  f"若刚改过 getSchema()，请先 drop_collection() 再重建。")
        try:
            self._db_client.load_collection(collection_name=self.col_name)
            print(self._db_client.get_load_state(collection_name=self.col_name))
        except Exception as e:
            print(f"[warn] load_collection 跳过（Milvus Lite 上是 no-op）: {e}")


    def getSchema(self):
        schema=self._db_client.create_schema()
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
         )
        schema.add_field(field_name="state_vec",datatype=DataType.FLOAT_VECTOR,dim=self.p.emb_dim)
        schema.add_field(field_name="state",datatype=DataType.VARCHAR,max_length=MAX_STATE_LEN)
        schema.add_field(field_name="action",datatype=DataType.VARCHAR,max_length=MAX_STATE_LEN)
        schema.add_field(field_name="action_vec",datatype=DataType.FLOAT_VECTOR,dim=self.p.emb_dim)
        schema.add_field(field_name="reward",datatype=DataType.DOUBLE)
        schema.add_field(field_name="g",datatype=DataType.DOUBLE)
        schema.add_field(field_name="episode_id",datatype=DataType.INT64)
        schema.add_field(field_name="step_index",datatype=DataType.INT64)
        schema.add_field("done",datatype=DataType.BOOL)
        schema.add_field("task",datatype=DataType.VARCHAR,max_length=1024)
        schema.add_field(field_name="task_vec",datatype=DataType.FLOAT_VECTOR,dim=self.p.emb_dim)
        #这里做了更改，task是一个对用户需求的总结字段，可以使用相似度进行检索
        return schema
    def getIndexParam(self):
        index_param=self._db_client.prepare_index_params()
        index_param.add_index(field_name="action_vec",index_type="AUTOINDEX",metric_type="COSINE")
        index_param.add_index(field_name="state_vec",index_type="AUTOINDEX",metric_type="COSINE")
        index_param.add_index(field_name="task_vec",index_type="AUTOINDEX",metric_type="COSINE")
        return index_param
    def get_embVector(self,text:str):
        if text in self.emb_cache:
            return self.emb_cache[text]
        resp=self._emb_client.embeddings.create(
            model=self.p.emb_model,        # 不要写死模型名，schema 的 dim 是跟着它走的
            input=text,
            encoding_format="float"
        )
        vec =  resp.data[0].embedding
        if len(vec) != self.p.emb_dim:
            raise ValueError(
                f"嵌入维度不匹配：模型 {self.p.emb_model} 返回 {len(vec)} 维，"
                f"但 emb_dim / Milvus schema 是 {self.p.emb_dim} 维"
            )
        self.emb_cache[text]=vec
        return vec
    def drop_collection(self):
        """删掉整个集合。⚠️ 改过 getSchema() 之后必须调它再重建，
        否则新字段不会生效（Milvus 不支持修改已有字段）。"""
        if self._db_client.has_collection(collection_name=self.col_name):
            self._db_client.drop_collection(collection_name=self.col_name)
            self.emb_cache.clear()
            print(f"已删除集合 {self.col_name}")
        else:
            print(f"集合 {self.col_name} 不存在，无需删除")
    def count(self)->int:
        """库里有多少条记忆。排查"召回为空"时先看它：
        为 -1 是查询失败；为 0 说明还没写进去；
        不为 0 但召回为空，那就是 radius 太严或 state 摘要粒度不对。"""
        try:
            rows=self._db_client.query(
                collection_name=self.col_name,
                filter="id >= 0",
                output_fields=["id"],
                limit=16384,
            )
            return len(rows)
        except Exception as e:
            print(f"[warn] count 失败: {e}")
            return -1
    def _generate_filter(self,f:list):
        f=[i for i in f if i is not None]
        if  f:
            filter=f"id in [{",".join(str(i["id"]) for i in f)}]"
            return filter
        else :
            return "id < 0"
    def _get_neighborhood(self,task:str,state:str,limit:int)->list:
        
        task_res=self._db_client.search(
            collection_name=self.col_name,
            data=[self.get_embVector(task)],
            anns_field="task_vec",
            limit=limit*5,
            search_params={
                "metric_type":"COSINE",
                "params":{"radius":0.5,"range_filter":1.0}
            },
            output_fields=["state"]
        )
        matched=[h for hits in task_res for h in hits ]
        id_filter=self._generate_filter(matched) 

        state_res=self._db_client.search(
            collection_name=self.col_name,
            data=[self.get_embVector(state)],
            anns_field="state_vec",
            limit=limit,
            filter=id_filter,
            search_params={
                "metric_type":"COSINE",
                "params":{"radius":0.8,"range_filter":1.0}
            },
            output_fields=["task","state","action","g"]
        )
        return [h for hits in state_res for h in hits]

    def _get_V(self,neighborhood:list)->float:
        if not neighborhood:
            return 0
        sum:DataType.DOUBLE=0
        for neighbor in neighborhood:
            sum=sum+neighbor["g"]
        V=sum/len(neighborhood)
        return V
    def get_A(self,task:str,state:str,actions:list[str],neighbood_limit:int=10000)->list[float]:
        neighbood=self._get_neighborhood(
            task=task,
            state=state,
            limit=neighbood_limit
        )
        id_filter=self._generate_filter(neighbood)
        V=self._get_V(neighbood)
        Q=self._get_Q(actions=actions,id_filter=id_filter)
        A=[q-V for q in Q]
        return A
    def _get_Q(self,actions:list,id_filter:str,limit:int = 10)->float:
        if not actions:
            return []
        action_res=self._db_client.search(
            collection_name=self.col_name,
            data=[self.get_embVector(a) for a in actions],
            anns_field="action_vec",
            limit=limit,
            filter=id_filter,
            search_params={
                "metric_type":"COSINE",
                "params":{"radius":0.8,"range_filter":1}
                
            },
            output_fields=["action","g"]
        )
        Q:list[float]=[]
        for hits in action_res:
            sum:float=0
            if not hits:
                Q.append(sum)
                continue
            for h in hits:
                sum=sum+h["g"]
            Q.append(sum/len(hits))

        return Q
            





    
        
        








