import os
import asyncio
import nest_asyncio
import json
import logging
import pandas as pd
import re
import httpx  # 需要导入这个库来控制网络请求
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightrag-server")

nest_asyncio.apply()

app = FastAPI(title="LightRAG Service for Evaluation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 核心配置区域 ====================

CORPUS_DIR = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（四）\rag\Datasets\Corpus"
DATASET_NAME = os.getenv("DATASET", "novel").lower()
WORKING_DIR = f"./my_rag_storage1/{DATASET_NAME.capitalize()}"

MODEL_NAME = "deepseek-chat"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-f2e6433f917d47a7b9a1cc188f65fd70")

# ====================================================

rag_instance = None

# ==================== 关键修改：增强网络稳定性 ====================
# 1. 设置极长的超时时间 (600秒 = 10分钟)，防止 DeepSeek 响应慢时报错
# 2. 限制最大连接数 (max_connections=5)，防止一次发太多请求把网络挤爆
timeout_config = httpx.Timeout(600.0, connect=60.0)
limits_config = httpx.Limits(max_keepalive_connections=5, max_connections=5)

openai_client = AsyncOpenAI(
    api_key=LLM_API_KEY, 
    base_url=LLM_BASE_URL,
    timeout=timeout_config,
    http_client=httpx.AsyncClient(limits=limits_config, timeout=timeout_config)
)
# ================================================================

async def llm_model_func(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in history_messages:
        messages.append(msg)
    messages.append({"role": "user", "content": prompt})
    
    try:
        # 增加重试逻辑，如果单次调用失败，让 LightRAG 知道这不是致命错误
        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed (will retry if possible): {e}")
        raise e 

def split_text_into_chunks(text, max_chars=1000):
    """手动将超长文本切分为小段落"""
    chunks = []
    paragraphs = text.split('\n')
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        
        if len(p) > max_chars:
            sentences = re.split(r'(?<=[.!?。？！])\s*', p)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) > max_chars:
                    chunks.append(current_chunk)
                    current_chunk = sent
                else:
                    current_chunk += sent
            if current_chunk:
                chunks.append(current_chunk)
        else:
            chunks.append(p)
            
    return chunks

async def ingest_data():
    """读取数据并手动预处理"""
    logger.info(f"📂 正在从源目录查找数据: {CORPUS_DIR}")
    
    json_path = os.path.join(CORPUS_DIR, f"{DATASET_NAME}.json")
    texts_to_insert = []

    if os.path.exists(json_path):
        logger.info(f"📖 发现 JSON 文件: {json_path}，正在读取...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_text = ""
            if isinstance(data, list):
                for item in data:
                    raw_text += json.dumps(item, ensure_ascii=False) + "\n"
            elif isinstance(data, dict):
                raw_text = data.get('context') or data.get('text') or json.dumps(data, ensure_ascii=False)
            
            logger.info(f"🔪 原始文本读取完毕 (长度 {len(raw_text)} 字符)，正在手动切片...")
            
            # 手动切片
            texts_to_insert = split_text_into_chunks(raw_text, max_chars=2000)
            
            logger.info(f"✅ 切片完成！共生成 {len(texts_to_insert)} 个小文档片段。")

        except Exception as e:
            logger.error(f"❌ 读取 JSON 失败: {e}")
            return
    else:
        logger.error(f"❌ 未找到文件: {json_path}")
        return

    if texts_to_insert:
        logger.info(f"🚀 准备将 {len(texts_to_insert)} 条数据片段插入索引...")
        logger.info("⏳ 正在继续构建知识图谱 (已完成的部分会自动跳过)...")
        try:
            await rag_instance.insert(texts_to_insert)
            logger.info("🎉🎉🎉 数据索引构建完成！恭喜！ 🎉🎉🎉")
        except Exception as e:
            logger.error(f"❌ 插入过程中发生错误: {e}")
    else:
        logger.warning("⚠️ 未提取到有效数据")

async def init_rag():
    global rag_instance
    logger.info(f"👉 目标数据集: {DATASET_NAME}")
    logger.info(f"👉 索引存储目录: {WORKING_DIR}")
    
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    logger.info("正在检查/加载嵌入模型...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
    
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embed(texts, tokenizer, embed_model),
    )
    
    rag_instance = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_name=MODEL_NAME,
        embedding_func=embedding_func,
        chunk_token_size=512,
        llm_model_kwargs={}
    )
    
    await rag_instance.initialize_storages()
    await initialize_pipeline_status()
    await ingest_data()

@app.on_event("startup")
async def startup_event():
    await init_rag()

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"

@app.post("/query")
async def query_rag(request: QueryRequest):
    if not rag_instance:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        param = QueryParam(mode=request.mode)
        ans_response = rag_instance.query(request.query, param=param)
        
        ctx_param = QueryParam(mode=request.mode, only_need_context=True)
        ctx_response = rag_instance.query(request.query, param=ctx_param)

        while asyncio.iscoroutine(ans_response): ans_response = await ans_response
        while asyncio.iscoroutine(ctx_response): ctx_response = await ctx_response
            
        return {
            "answer": str(ans_response),
            "context": [str(ctx_response)]
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {
        "dataset": DATASET_NAME,
        "working_dir": WORKING_DIR,
        "corpus_source": CORPUS_DIR,
        "status": "ready"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)