# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TeslaCar_QA is a RAG (Retrieval-Augmented Generation) based question-answering system for Tesla Model 3 user manual. The system extracts, indexes, and retrieves relevant information from PDF documentation to answer user queries using LLM-powered generation.

## Key Architecture

### Data Flow Pipeline
1. **Document Processing** (`src/parser/`): PDF extraction → Text cleaning → Semantic chunking → MongoDB storage
2. **Retrieval** (`src/retriever/`): Multi-retriever recall (BM25, Milvus, FAISS, TF-IDF) → Document merging
3. **Reranking** (`src/reranker/`): BGE-M3 or Qwen3 rerankers refine top-k candidates
4. **Generation** (`src/client/`): LLM generates answers with citations from retrieved context
5. **Evaluation** (`final_score.py`): Semantic similarity + keyword matching + RAGAs metrics

### Core Components

**Document Processing**
- `src/parser/pdf_parse.py`: Extracts text/images from PDF using PyMuPDF, generates unique IDs (MD5), creates Document objects
- `src/parser/image_handler.py`: Extracts and saves images with metadata
- `src/fields/manual_info_mongo.py`: Pydantic model for document metadata storage
- `src/fields/manual_images.py`: Pydantic model for image metadata (page, path, title)

**Retrieval Layer**
- All retrievers inherit from `BaseRetriever` with `retrieve_topk(query, topk)` method
- Implementations: BM25 (jieba tokenization), Milvus (BGE-M3 dense+sparse), FAISS, TF-IDF, Qwen3
- Retrievers support two modes: build index (`retrieve=False`) or load existing (`retrieve=True`)

**Reranking Layer**
- All rerankers inherit from `BaseReranker` with `rank(query, docs, topk)` method
- Implementations: BGE-M3 (cross-encoder), Qwen3 (local or vLLM-based)

**LLM Clients**
- `src/client/llm_hyde_client.py`: HyDE (Hypothetical Document Embeddings) query expansion using DeepSeek API
- `src/client/llm_local_client.py`: Local inference via vLLM service (default port 8000)
- `src/client/llm_chat_client.py`: External LLM APIs (DOUBAO/DeepSeek/GPT-4) for data generation
- `src/client/llm_clean_client.py`: Text cleaning via LLM API
- `src/client/semantic_chunk_client.py`: Calls semantic chunking service (port 6000)
- `src/client/mongodb_config.py`: MongoDB connection manager

**Data Generation**
- `src/gen_qa/run.py`: Multi-threaded QA pair generation from documents
  - `gen_qa()`: Concurrent generation with checkpoint recovery
  - `chat()`: API calls with retry mechanism
  - Templates: CONTEXT_PROMPT_TPL (QA generation), GENERALIZE_PROMPT_TPL (question variants), KEYWORDS_PROMPT_TPL (keyword extraction)

### Configuration Management

**Path Constants** (`src/constant.py`)
```python
base_dir = "D:/Project/"  # Modify per environment
pdf_path = base_dir + "data/Tesla_Manual.pdf"
raw_docs_path = base_dir + "data/processed_docs/raw_docs.pkl"
clean_docs_path = base_dir + "data/processed_docs/clean_docs.pkl"
split_docs_path = base_dir + "data/processed_docs/split_docs.pkl"
# Plus: stopwords_path, image_save_dir, bm25_pickle_path, milvus_db_path, model paths
```

**Environment Variables** (Required)
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL_NAME`: For HyDE and cleaning
- `DOUBAO_API_KEY`, `DOUBAO_BASE_URL`, `DOUBAO_MODEL_NAME`: For evaluation and data generation
- MongoDB connection settings (if externalized)

## Common Commands

### Environment Setup
```bash
# Create environment (Python 3.8+ recommended)
conda create -n rag python=3.8
conda activate rag

# Install dependencies
pip install -r requirements.txt
```

### Service Management

**Start MongoDB**
```bash
mongodb-7.0.20/bin/mongod --port=27017 --dbpath=data/mongodb --fork
```

**Start Semantic Chunking Service**
```bash
python src/server/semantic_chunk.py  # Runs on port 6000
```

**Start vLLM Inference Service**
```bash
# For base model
vllm serve models/Qwen2.5-7B-Instruct --max-model-len 8192 --port 8000

# For fine-tuned model
vllm serve output/tesla_qwen_int4 --max-model-len 8192 --port 8000
```

### Index Building
```bash
python build_index.py
```
Pipeline: `load_pdf()` → `request_llm_clean()` → `texts_split()` → `BM25()` + `MilvusRetriever()`

### Inference
```bash
python infer.py
```
Interactive loop: User query → Multi-retriever recall → Merge → Rerank → LLM generation → Citation formatting

### Data Generation & Fine-tuning

**Generate QA Pairs**
```bash
python src/gen_qa/run.py
```
Outputs: `qa_pair.json`, `expand_qa.json`, `keywords.json`, `train_qa_pair.json`, `test_qa_pair.json`

**Generate SFT Dataset**
```bash
python generate_sft_dataset.py
```
Outputs: `data/summary_data/train.json`, `data/rerank_data/train.json`

**Fine-tune with LLaMA-Factory**
```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/tesla_qwen_lora.yaml
llamafactory-cli export examples/merge_lora/tesla_qwen_lora.yaml
```

**Quantization (Optional)**
```bash
python awq_quant.py --model_path output/tesla_qwen_merged --output_path output/tesla_qwen_int4
```

### Evaluation
```bash
python final_score.py
```
Metrics: Jaccard keyword matching, semantic similarity (text2vec), RAGAs (LLMContextRecall, LLMContextPrecisionWithReference)

## Important Implementation Details

### Document Chunking Strategy
- **Parent Chunks**: Max 512 tokens via semantic clustering (AgglomerativeClustering)
- **Child Chunks**: 256 tokens with 50 token overlap (RecursiveCharacterTextSplitter)
- Stored in MongoDB with unique_id, page_content, metadata (page, source, parent_id)

### Multi-Retriever Fusion
- BM25: Keyword-based, jieba tokenization, pickle cached
- Milvus: Dense (768-dim) + Sparse vectors from BGE-M3, RRF (Reciprocal Rank Fusion)
- FAISS: Dense-only vector search
- Results merged with `merge_docs()` to deduplicate by page_content

### Answer Format
LLM must output: `{答案}【{引用编号1},{引用编号2},...】`
- `post_processing()` extracts answer text and citation numbers
- Citations mapped back to source documents for verification

### LLM Prompt Structure
```python
LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统...
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{答案}【{引用编号1},{引用编号2},...】
如果无法从中得到答案，请说 "无答案"。
"""
```

### Utility Functions (`src/utils.py`)
- `merge_docs(docs1, docs2)`: Deduplicates documents by page_content
- `post_processing(response, ranked_docs)`: Parses answer and extracts citations

## Development Workflow

1. **Modify paths** in `src/constant.py` to match your environment
2. **Set environment variables** for API keys (DEEPSEEK, DOUBAO)
3. **Start services** (MongoDB, semantic_chunk, vLLM) in separate terminals
4. **Build indexes** once per document corpus change
5. **Test retrieval** via `infer.py` before fine-tuning
6. **Generate data** → **Fine-tune** → **Re-deploy vLLM** → **Evaluate**
7. **Iterate** on hyperparameters (topk values, reranker choice, prompt templates)

## Key Dependencies

- **LLM Inference**: vllm, torch, transformers
- **Embeddings**: sentence-transformers, FlagEmbedding (BGE-M3)
- **Vector Stores**: milvus-lite, faiss-cpu
- **Text Processing**: langchain, langchain-text-splitters, jieba
- **PDF Parsing**: PyMuPDF (fitz)
- **Evaluation**: ragas, text2vec
- **Fine-tuning**: LLaMA-Factory (separate submodule)

## External Dependencies

- **RAG-Retrieval**: Submodule for advanced retrieval/reranking models (currently in .gitignore)
- **LLaMA-Factory**: Submodule for LLM fine-tuning (LoRA, full-parameter, quantization)
- Clone separately:
  ```bash
  git clone https://github.com/NLPJCL/RAG-Retrieval.git
  git clone https://github.com/hiyouga/LLaMA-Factory.git
  ```

## Testing Notes

- **Unit tests**: Should be added for retriever/reranker modules
- **Integration tests**: Test full pipeline with sample queries
- **Evaluation**: Use `final_score.py` with test set (10% split from generated QA pairs)
- **Target metrics**: Average score > 0.7 (0.2 * keyword + 0.8 * semantic)

## Troubleshooting

- **MongoDB connection errors**: Check port 27017, ensure --dbpath exists
- **vLLM OOM**: Reduce --max-model-len or use quantized models
- **Slow retrieval**: Check if indexes are prebuilt (pickle/Milvus DB exist)
- **Empty results**: Verify PDF page range (_min_filter_pages=4, _max_filter_pages=247)
- **API rate limits**: Add retries in `src/gen_qa/run.py` chat() function
