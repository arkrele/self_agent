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
_ENTITY_FIELDS: tuple[str, ...] = ("state","action","reward","g","episode_id","step_index")

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

class jitRL_DBParams(BaseModel):
    db_path:str
    emb_url:str
    emb_key:str
    col_name:str
    emb_model:str = "BAAI/bge-m3"
    emb_dim:int = 1024

@dataclass
class Neighbor():
    state:str
    action:str
    g:float
    reward:float
    similarity:float
    episode_id:int
    step_index:int

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
        return schema
    def getIndexParam(self):
        index_param=self._db_client.prepare_index_params()
        index_param.add_index(field_name="action_vec",index_type="AUTOINDEX",metric_type="COSINE")
        index_param.add_index(field_name="state_vec",index_type="AUTOINDEX",metric_type="COSINE")
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
    def add_episode(self,
                episode_id: int,
                steps: list[tuple[str, str, float]],
                gamma: float,
                done: bool = True):
        """一个 episode 走完后批量落库，return-to-go 在这里算：

            G_t = Σ_{u≥t} γ^(u−t) · r_u

        steps: [(state_summary, action, reward), ...]，必须按时间顺序。
        ⚠️ gamma 在写入时固定，换 gamma 需要重刷数据。
        """
        n = len(steps)
        if n == 0:
            return 0

        records: list[RecordModel] = []
        for t, (state, action, reward) in enumerate(steps):
            if len(state) > MAX_STATE_LEN:
                raise ValueError(
                    f"step {t} 的 state 长 {len(state)} 字符，超过 VARCHAR 上限 {MAX_STATE_LEN}。"
                    f"更严重的是：存原始 observation 会让每条都独一无二、互相相似度极低，"
                    f"邻居召回永远为空、优势恒为 0、记忆静默失效。请先做摘要。"
                )
            g = sum(gamma ** (k - t) * steps[k][2] for k in range(t, n))
            records.append(RecordModel(
                state=state,
                action=action,
                reward=float(reward),
                g=float(g),
                episode_id=int(episode_id),
                step_index=int(t),
                done=bool(done and t == n - 1),   # 只有最后一条是 True
                # state_vec / action_vec 留 None，下面补
            ))

        for r in records:
            if r.state_vec is None:      # 用 is None，不用 not（空列表会被 not 判成没算过）
                r.state_vec = self.get_embVector(r.state)
            if r.action_vec is None:
                r.action_vec = self.get_embVector(r.action)

        self._db_client.insert(
            collection_name=self.col_name,
            data=[r.model_dump() for r in records],
        )
        return len(records)

    def query_neighbors(self,state:str,limit:int=50,radius:float=0.5)->list[Neighbor]:
        """只按 state_vec 召回邻居，**不做任何动作过滤**。

        这是 JitRL 唯一需要的检索接口：
            Q̂(s,a) = mean{G | 邻居中 action == a}
            V̂(s)   = mean{所有邻居的 G}      ← 基线必须看到全体邻居
        加了 action 过滤会让 V̂ 偏移甚至返回空。

        ⚠️ 入参 state 必须是**摘要**，且与 add_episode 写入时用的是同一个摘要函数。
           传原始 observation 会永远召回空 → Â≡0 → 记忆静默失效。
        """
        res = self._db_client.search(
            collection_name=self.col_name,
            data=[self.get_embVector(state)],
            anns_field="state_vec",
            limit=limit,
            search_params={
                "metric_type": "COSINE",
                # Milvus 对 COSINE 返回 radius < sim <= range_filter
                "params": {"radius": radius, "range_filter": 1.0},
            },
            output_fields=_OUTPUT_FIELDS,
        )
        return [self._to_neighbor(h) for hits in res for h in hits]

    @staticmethod
    def _to_neighbor(hit)->Neighbor:
        """把 Milvus 的 hit 解成 Neighbor。

        MilvusClient.search 返回 list[list[dict]]，hit 形如
            {"id":…, "distance":…, "entity": {"state":…, "action":…, "g":…}}
        老式 Collection.search 返回对象，这里一并兼容。
        """
        e = _hit_get(hit, "entity", None) or {}
        if not isinstance(e, dict):
            e = {k: getattr(e, k, None) for k in _ENTITY_FIELDS}

        def pick(name, default):
            v = e.get(name, default)
            return default if v is None else v

        return Neighbor(
            state=str(pick("state", "")),
            action=str(pick("action", "")),
            g=float(pick("g", 0.0)),
            reward=float(pick("reward", 0.0)),
            similarity=float(_hit_get(hit, "distance", 0.0) or 0.0),
            episode_id=int(pick("episode_id", -1)),
            step_index=int(pick("step_index", -1)),
        )

    def query_record(self,state:str,action:str,
                     state_topk:int=20,action_topk:int=5)->list[Neighbor]:
        """先按 state 召回，再从中挑 action 相似的记录。

        ⚠️ 这个接口**不能**用来算 V̂ 基线（基线必须看到全部邻居），
           基线走 query_neighbors。这里适合"查某个动作在相似状态下的表现"。
        """
        state_res=self._db_client.search(
            collection_name=self.col_name,
            data=[self.get_embVector(state)],
            anns_field="state_vec",
            limit=state_topk,
            search_params={
                "metric_type":"COSINE",
                "params":{"radius":0.5,"range_filter":1.0},  # 相似度 >= 0.5
            },
            output_fields=["state"]
        )
        matched_ids=[_hit_get(h,"id") for hits in state_res for h in hits]
        matched_ids=[i for i in matched_ids if i is not None]
        if not matched_ids:
            return []            # 空的时候 "id in []" 是非法表达式，Milvus 会报错
        id_filter=f"id in [{','.join(str(i) for i in matched_ids)}]"
        action_res=self._db_client.search(
            collection_name=self.col_name,
            data=[self.get_embVector(action)],
            anns_field="action_vec",
            limit=action_topk,
            filter=id_filter,
            search_params={
                "metric_type":"COSINE",
                "params":{"radius":0.8,"range_filter":1.0}
            },
            output_fields=_OUTPUT_FIELDS  
        )
        return [self._to_neighbor(h) for hits in action_res for h in hits]
