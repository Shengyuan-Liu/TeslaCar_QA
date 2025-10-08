# Tesla RAG 问答系统开发流程

本文档提供从零开始实现该项目的完整流程指南。

---

## 阶段一：环境准备（第1-2天）

### 1.1 项目初始化
```bash
mkdir -p {src/{parser,retriever,reranker,client,fields,server,gen_qa},data,models,log}
conda create -n rag python=3.8 && conda activate rag
```

### 1.2 依赖安装
```bash
pip install -r requirements.txt
```

**检查点**：✅ 环境可用，基础库导入正常

---

## 阶段二：文档处理管道（第3-5天）

### 2.1 PDF解析模块
**文件**：`src/parser/pdf_parse.py`

**核心函数**：
- `load_pdf() -> list[Document]`
  - 打开PDF，遍历页面
  - 提取文本和图片
  - 生成 unique_id (MD5)
  - 返回 Document 列表（page_content + metadata）

**类/结构**：
- `Document`: Langchain文档对象（page_content, metadata）

### 2.2 图片处理（可选）
**文件**：`src/parser/image_handler.py`

**核心函数**：
- `handle_image(img, img_index, page) -> ManualImages`
  - 提取图片二进制数据
  - 保存到 `data/saved_images/`
  - 返回图片元数据

### 2.3 语义切分服务
**文件**：`src/server/semantic_chunk.py`

**核心类/函数**：
- FastAPI应用（端口6000）
- `POST /v1/semantic-chunks`
  - 输入：sentences, group_size
  - SentenceTransformer 计算嵌入
  - AgglomerativeClustering 聚类
  - 返回合并后的chunks

**启动**：`python src/server/semantic_chunk.py`

### 2.4 语义切分客户端
**文件**：`src/client/semantic_chunk_client.py`

**核心函数**：
- `request_semantic_chunk(text, group_size) -> list[str]`
  - 调用语义切分服务API
  - 返回分组后的文本列表

### 2.5 文本清洗
**文件**：`src/client/llm_clean_client.py`

**核心函数**：
- `request_llm_clean(raw_docs) -> list[Document]`
  - 调用外部LLM API清洗文本
  - 去除乱码、修正格式

### 2.6 文档切分
**文件**：`src/parser/pdf_parse.py`

**核心函数**：
- `texts_split(raw_docs) -> list[Document]`
  - 调用语义切分服务
  - 生成父chunk（≤512 tokens）
  - RecursiveCharacterTextSplitter生成子chunk（256 tokens, overlap 50）
  - 保存到MongoDB（`save_2_mongo`）
  - 返回所有切分文档

**检查点**：✅ PDF解析成功，文档切分合理（500-2000个chunk）

---

## 阶段三：存储与索引（第6-8天）

### 3.1 MongoDB配置
**文件**：`src/client/mongodb_config.py`

**核心类**：
- `MongoConfig`
  - `get_collection(name) -> Collection`

### 3.2 数据模型
**文件**：`src/fields/manual_info_mongo.py`

**核心类**：
- `ManualInfo(BaseModel)`
  - unique_id: str
  - page_content: str
  - metadata: dict

**文件**：`src/fields/manual_images.py`

**核心类**：
- `ManualImages(BaseModel)`
  - 图片元数据

### 3.3 检索器基类
**文件**：`src/retriever/retriever.py`

**核心类**：
- `BaseRetriever(ABC)`
  - `retrieve_topk(query, topk) -> list[Document]` (抽象方法)

### 3.4 BM25检索器
**文件**：`src/retriever/bm25_retriever.py`

**核心类**：
- `BM25(BaseRetriever)`
  - `__init__(docs, retrieve=False)`
    - jieba分词
    - 构建BM25索引或加载pickle
  - `retrieve_topk(query, topk) -> list[Document]`

### 3.5 Milvus检索器
**文件**：`src/retriever/milvus_retriever.py`

**核心类**：
- `MilvusRetriever(BaseRetriever)`
  - `__init__(docs, retrieve=False)`
    - 加载BGE-M3模型
    - 创建Milvus collection（稠密+稀疏向量）
    - 插入文档或加载已有数据库
  - `retrieve_topk(query, topk) -> list[Document]`
    - 稠密检索 + 稀疏检索
    - RRF融合

### 3.6 FAISS检索器
**文件**：`src/retriever/faiss_retriever.py`

**核心类**：
- `FaissRetriever(BaseRetriever)`
  - `__init__(docs, retrieve=False)`
    - HuggingFaceEmbeddings
    - FAISS.from_documents()
  - `retrieve_topk(query, topk)`

### 3.7 TF-IDF检索器
**文件**：`src/retriever/tfidf_retriever.py`

**核心类**：
- `TFIDF(BaseRetriever)`
  - TF-IDF向量化
  - `retrieve_topk(query, topk)`

### 3.8 Qwen3检索器
**文件**：`src/retriever/qwen3_retriever.py`

**核心类**：
- `Qwen3Retriever(BaseRetriever)`
  - Qwen3 Embedding模型
  - `retrieve_topk(query, topk)`

**检查点**：✅ 各检索器能返回相关文档

---

## 阶段四：LLM推理服务（第9-11天）

### 4.1 下载模型
```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir models/Qwen2.5-7B
```

### 4.2 启动vLLM服务
```bash
vllm serve models/Qwen2.5-7B-Instruct --max-model-len 8192 --port 8000
```

### 4.3 本地LLM客户端
**文件**：`src/client/llm_local_client.py`

**核心函数**：
- `request_chat(query, context, stream=False)`
  - OpenAI客户端连接本地vLLM
  - 构建prompt（query + context）
  - 返回生成的答案

### 4.4 HyDE客户端（可选）
**文件**：`src/client/llm_hyde_client.py`

**核心函数**：
- `request_hyde(query) -> str`
  - 生成假设性答案
  - 用于查询增强

### 4.5 外部LLM客户端
**文件**：`src/client/llm_chat_client.py`

**核心函数**：
- `request_chat(query, context)`
  - 调用DOUBAO/DeepSeek/GPT-4 API
  - 用于评估和数据生成

**检查点**：✅ vLLM服务正常，能生成回答

---

## 阶段五：端到端打通（第12-14天）

### 5.1 工具函数
**文件**：`src/utils.py`

**核心函数**：
- `merge_docs(docs1, docs2) -> list[Document]`
  - 基于page_content去重
- `post_processing(response, ranked_docs) -> dict`
  - 解析答案和引用编号
  - 返回 {"answer": "...", "citations": [...]}

### 5.2 路径配置
**文件**：`src/constant.py`

**核心变量**：
- `base_dir`: 项目根目录
- `pdf_path`, `stopwords_path`
- `bm25_pickle_path`, `milvus_db_path`
- 各种模型路径

### 5.3 索引构建脚本
**文件**：`build_index.py`

**流程**：
```python
# 1. 加载或解析PDF
raw_docs = load_pdf() or pickle.load(raw_docs_path)

# 2. 清洗文本
clean_docs = request_llm_clean(raw_docs) or pickle.load(clean_docs_path)

# 3. 切分文档
split_docs = texts_split(clean_docs) or pickle.load(split_docs_path)

# 4. 构建索引
bm25_retriever = BM25(split_docs)
milvus_retriever = MilvusRetriever(split_docs)

# 5. 测试召回
candidate_docs = bm25_retriever.retrieve_topk("测试query", topk=3)
```

### 5.4 推理主程序
**文件**：`infer.py`

**流程**：
```python
# 加载检索器和重排器
bm25_retriever = BM25(retrieve=True)
milvus_retriever = MilvusRetriever(retrieve=True)
bge_m3_reranker = BGEM3ReRanker(model_path)

# 主循环
while True:
    query = input("输入—>")

    # 召回
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=10)
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)

    # 去重
    merged_docs = merge_docs(bm25_docs, milvus_docs)

    # 精排
    ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=5)

    # 生成答案
    context = "\n".join([f"{idx+1}. {doc.page_content}" for idx, doc in enumerate(ranked_docs)])
    response = request_chat(query, context, stream=True)

    # 后处理
    answer = post_processing(response, ranked_docs)
```

**检查点**：✅ 能完整问答（虽然质量可能不高）

---

## 阶段六：效果优化（第15-20天）

### 6.1 重排序器基类
**文件**：`src/reranker/reranker.py`

**核心类**：
- `BaseReranker(ABC)`
  - `rank(query, docs, topk) -> list[Document]` (抽象方法)

### 6.2 BGE-M3重排器
**文件**：`src/reranker/bge_m3_reranker.py`

**核心类**：
- `BGEM3ReRanker(BaseReranker)`
  - `__init__(model_path)`
    - 加载FlagReranker模型
  - `rank(query, docs, topk) -> list[Document]`
    - 计算query-doc分数
    - 排序返回top-k

### 6.3 Qwen3重排器
**文件**：`src/reranker/qwen3_reranker.py`

**核心类**：
- `Qwen3ReRanker(BaseReranker)`
  - Qwen3 Reranker模型
  - `rank(query, docs, topk)`

**文件**：`src/reranker/qwen3_reranker_vllm.py`

**核心类**：
- `Qwen3ReRankervLLM(BaseReranker)`
  - 通过vLLM服务调用Qwen3重排
  - `rank(query, docs, topk)`

### 6.4 集成到推理流程
在 `infer.py` 的召回和生成之间添加：
```python
ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=5)
```

### 6.5 Prompt优化
在 `llm_local_client.py` 或 `llm_chat_client.py` 中定义：
```python
LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统...
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}},{{引用编号2}},...】
如果无法从中得到答案，请说 "无答案"。
"""
```

**检查点**：✅ 答案相关性提升，格式规范

---

## 阶段七：数据生成与微调（第21-30天）

### 7.1 自动生成QA数据
**文件**：`src/gen_qa/run.py`

**核心函数**：
- `build_qa_prompt(prompt_tmpl, text) -> str`
  - 填充提示词模板

- `chat(prompt, max_retry, temperature, top_p) -> str`
  - 调用外部LLM API
  - 重试机制

- `gen_qa(splitted_docs, prompt_tmpl, qa_ckpt_filename, expand) -> dict`
  - ThreadPoolExecutor并发生成
  - 逐行写入checkpoint文件
  - 返回生成结果字典

**流程**：
```python
# 1. 从文档生成QA对
qa_dict = gen_qa(splitted_docs, CONTEXT_PROMPT_TPL, "qa_pair.json")

# 2. 问题泛化（1个→5个变体）
question_docs = [Document(page_content=qa["question"]) for qa in qa_list]
expand_qa_dict = gen_qa(question_docs, GENERALIZE_PROMPT_TPL, "expand_qa.json", expand=True)

# 3. 提取关键词
test_answer_docs = [Document(page_content=answer) for answer in unique_answers]
keywords_dict = gen_qa(test_answer_docs, KEYWORDS_PROMPT_TPL, "keywords.json", expand=True)

# 4. 添加负样本
chats_data = open("raw_general_chats.txt").readlines()
for line in chats_data:
    train_qa_pairs.append({"question": line, "answer": "无答案"})

# 5. 拆分训练/测试集（9:1）
random.shuffle(all_qa_pairs)
train = all_qa_pairs[:int(len(all_qa_pairs)*0.9)]
test = all_qa_pairs[int(len(all_qa_pairs)*0.9):]

# 6. 保存
json.dump(train, open("train_qa_pair.json", "w"))
json.dump(test, open("test_qa_pair.json", "w"))
```

**Prompt模板**：
- `CONTEXT_PROMPT_TPL`: 从文档生成QA
- `GENERALIZE_PROMPT_TPL`: 问题泛化
- `KEYWORDS_PROMPT_TPL`: 提取关键词
- `QA_QUALITY_PROMPT_TPL`: 质量评分

### 7.2 生成SFT训练数据
**文件**：`generate_sft_data.py`

**流程**：
```python
# 1. 对每个QA对运行检索+重排
for item in qa_pairs:
    query = item["question"]
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    ranked_docs = reranker.rank(query, merged_docs, topk=5)

    # 2. 生成答案
    response = request_chat(query, context)

    # 3. 格式化为训练数据
    # Summary数据：{"instruction": prompt, "input": "", "output": "答案【引用】"}
    # Rerank数据：{"query": query, "content": doc, "label": 0/1/2}

# 4. 保存
json.dump(summary_train, open("data/summary_data/train.json", "w"))
json.dump(rerank_train, open("data/rerank_data/train.json", "w"))
```

### 7.3 微调Qwen3模型
**目录**：`LLaMA-Factory-main/`

**配置文件**：
- `data/dataset_info.json`: 注册数据集
- `examples/train_lora/tesla_qwen_lora.yaml`: 训练配置

**命令**：
```bash
cd LLaMA-Factory-main
llamafactory-cli train examples/train_lora/tesla_qwen_lora.yaml
```

### 7.4 合并LoRA权重
```bash
llamafactory-cli export examples/merge_lora/tesla_qwen_lora.yaml
```

### 7.5 量化（可选）
```bash
python awq_quant.py --model_path output/tesla_qwen_merged --output_path output/tesla_qwen_int4
```

### 7.6 重启vLLM服务
```bash
vllm serve output/tesla_qwen_int4 --max-model-len 8192
```

**检查点**：✅ 微调后答案质量、格式更好

---

## 阶段八：评估与优化（第31-35天）

### 8.1 评估脚本
**文件**：`final_score.py`

**流程**：
```python
# 1. 加载检索器和重排器
bm25_retriever = BM25(retrieve=True)
milvus_retriever = MilvusRetriever(retrieve=True)
bge_m3_reranker = BGEM3ReRanker(model_path)
simModel = SentenceModel(text2vec_model_path)

# 2. 对测试集运行推理
for item in test_qa_pairs:
    query = item["question"]
    # 检索+重排+生成
    pred_answer = run_inference_pipeline(query)
    item["pred"] = pred_answer

# 3. 计算评分
def calc_jaccard(pred_keywords, gold_keywords, threshold=0.3):
    score = len(set(pred_keywords) & set(gold_keywords)) / (len(gold_keywords) + 1e-6)
    return 1 if score > threshold else 0

def report_score(result):
    for item in result:
        gold = item["answer"]
        pred = item["pred"]["answer"]

        if gold == "无答案":
            score = 1.0 if pred == gold else 0.0
        else:
            # 语义相似度
            semantic_score = semantic_search(simModel.encode([gold]), simModel.encode([pred]))[0][0]['score']
            # 关键词匹配
            keyword_score = calc_jaccard(extract_keywords(pred), item["keywords"])
            # 加权
            score = 0.2 * keyword_score + 0.8 * semantic_score

        item["score"] = score
    return result

# 4. Ragas评估
llm = ChatOpenAI(model=os.environ["DOUBAO_MODEL_NAME"], ...)
dataset = [{"user_input": q, "retrieved_contexts": [ctx], "response": pred, "reference": gold}]
result = evaluate(dataset, metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()], llm=llm)

# 5. 输出结果
final_score = np.mean([item["score"] for item in results])
print(f"语义相似度+关键词得分: {final_score}")
print(f"RAGas综合得分: {result}")
```

**核心函数**：
- `calc_jaccard(list_a, list_b, threshold)`: 关键词匹配分数
- `report_score(result)`: 计算综合评分

### 8.2 超参数调优
```python
configs = [
    {"bm25_k": 5, "faiss_k": 10, "rerank_k": 5},
    {"bm25_k": 10, "faiss_k": 10, "rerank_k": 5},
    {"bm25_k": 5, "faiss_k": 15, "rerank_k": 7},
]

for cfg in configs:
    score = batch_evaluate(cfg)
    print(f"Config {cfg}: Score {score}")
```

**检查点**：✅ 平均分 > 0.7

---

## 阶段九：工程化封装（第36-40天）

### 9.1 配置管理
完善 `src/constant.py`，支持环境变量：
```python
BASE_DIR = os.getenv("RAG_BASE_DIR", "/root/autodl-tmp/RAG/")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
```

### 9.2 服务启动脚本
**文件**：`config.ini` 或 `start_services.sh`

**流程**：
```bash
#!/bin/bash
# 1. 启动MongoDB
mongodb-7.0.20/bin/mongod --port=27017 --dbpath=data/mongodb --fork

# 2. 启动语义切分服务
nohup python src/server/semantic_chunk.py > log/semantic_chunk.log 2>&1 &

# 3. 启动vLLM服务
nohup vllm serve output/tesla_qwen_int4 --max-model-len 8192 > log/vllm.log 2>&1 &

# 4. 健康检查
curl http://localhost:8000/health
```

### 9.3 日志与异常处理
在关键模块添加日志：
```python
import logging
logger = logging.getLogger(__name__)

def retrieve_topk(query, topk):
    try:
        logger.info(f"开始检索: query={query}")
        results = self.search(query, topk)
        logger.info(f"检索完成: {len(results)}条")
        return results
    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        return []
```

### 9.4 测试脚本
为核心模块编写测试：
```python
# tests/test_retriever.py
def test_bm25():
    retriever = BM25(retrieve=True)
    results = retriever.retrieve_topk("自动上锁", topk=3)
    assert len(results) > 0

# tests/test_reranker.py
def test_bge_m3():
    reranker = BGEM3ReRanker(model_path)
    ranked = reranker.rank("自动上锁", docs, topk=5)
    assert len(ranked) == 5
```

**检查点**：✅ 服务一键启动，日志完善

---

## 阶段十：文档与交付（第41-42天）

### 10.1 编写文档
- `README.md`: 快速开始、安装指南
- `CLAUDE.md`: 架构说明（已完成）
- `Process.md`: 开发流程（本文档）
- `DEPLOYMENT.md`: 部署指南

### 10.2 代码审查清单
```markdown
✅ 所有路径使用 constant.py 管理
✅ 关键函数有docstring
✅ 异常有日志记录
✅ 敏感信息通过环境变量配置
✅ requirements.txt 完整
✅ 有测试数据和评估脚本
✅ 服务启动/停止脚本完善
```

---

## 核心文件与模块总览

### 根目录脚本
| 文件 | 功能 | 核心流程 |
|------|------|---------|
| `build_index.py` | 索引构建 | load_pdf → clean → split → build_index |
| `infer.py` | 推理主程序 | retrieve → merge → rerank → generate |
| `generate_sft_data.py` | SFT数据生成 | retrieve → generate → format → save |
| `final_score.py` | 评估 | inference → calc_score → ragas_eval |

### src/parser/
| 文件 | 核心类/函数 |
|------|-----------|
| `pdf_parse.py` | load_pdf(), texts_split(), save_2_mongo() |
| `image_handler.py` | handle_image() |

### src/retriever/
| 文件 | 核心类 | 方法 |
|------|--------|------|
| `retriever.py` | BaseRetriever | retrieve_topk() |
| `bm25_retriever.py` | BM25 | retrieve_topk() |
| `milvus_retriever.py` | MilvusRetriever | retrieve_topk() |
| `faiss_retriever.py` | FaissRetriever | retrieve_topk() |
| `tfidf_retriever.py` | TFIDF | retrieve_topk() |
| `qwen3_retriever.py` | Qwen3Retriever | retrieve_topk() |

### src/reranker/
| 文件 | 核心类 | 方法 |
|------|--------|------|
| `reranker.py` | BaseReranker | rank() |
| `bge_m3_reranker.py` | BGEM3ReRanker | rank() |
| `qwen3_reranker.py` | Qwen3ReRanker | rank() |
| `qwen3_reranker_vllm.py` | Qwen3ReRankervLLM | rank() |

### src/client/
| 文件 | 核心函数 | 用途 |
|------|---------|------|
| `llm_local_client.py` | request_chat() | vLLM推理 |
| `llm_chat_client.py` | request_chat() | 外部API评估 |
| `llm_hyde_client.py` | request_hyde() | HyDE查询增强 |
| `llm_clean_client.py` | request_llm_clean() | 文本清洗 |
| `semantic_chunk_client.py` | request_semantic_chunk() | 语义切分 |
| `mongodb_config.py` | MongoConfig.get_collection() | MongoDB连接 |

### src/server/
| 文件 | 核心组件 |
|------|---------|
| `semantic_chunk.py` | FastAPI服务, POST /v1/semantic-chunks |

### src/gen_qa/
| 文件 | 核心函数 |
|------|---------|
| `run.py` | gen_qa(), chat(), build_qa_prompt() |

### src/fields/
| 文件 | 核心类 |
|------|--------|
| `manual_info_mongo.py` | ManualInfo(BaseModel) |
| `manual_images.py` | ManualImages(BaseModel) |

### src/
| 文件 | 核心函数 |
|------|---------|
| `constant.py` | 所有路径和配置常量 |
| `utils.py` | merge_docs(), post_processing() |

---

## 开发原则

1. **增量开发**：每个模块先MVP，跑通后再优化
2. **及时测试**：完成一个模块立即测试
3. **Mock优先**：先用假数据验证流程
4. **日志驱动**：关键步骤打日志
5. **配置外置**：无硬编码路径

---

## 时间规划

| 阶段 | 天数 | 核心产出 | 验收标准 |
|------|------|---------|---------|
| 环境准备 | 2 | 项目结构、环境 | 能import所有库 |
| 文档处理 | 3 | PDF解析、切分 | 得到Document列表 |
| 存储索引 | 3 | MongoDB、检索器 | 检索有结果 |
| LLM服务 | 3 | vLLM部署、客户端 | 能生成回答 |
| 端到端 | 3 | build_index + infer | 完整问答 |
| 效果优化 | 6 | Reranker、Prompt | 答案质量可用 |
| 模型微调 | 10 | 数据生成、训练 | 格式规范 |
| 评估优化 | 5 | 评估脚本、调参 | 分数>0.7 |
| 工程化 | 5 | 配置、日志、脚本 | 一键部署 |
| 文档交付 | 2 | 完整文档 | 可复现 |

**总计：42天（6周）**
