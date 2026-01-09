import os
import asyncio
import nest_asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import logging
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightrag-server")

nest_asyncio.apply()

app = FastAPI(title="LightRAG Service")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

# Configuration (Hardcoded for the Medical corpus run)
WORKING_DIR = "./my_rag_storage/Medical"
MODEL_NAME = "deepseek-chat"
# Must match the model used during indexing (bge-large outputs 1024 dim)
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-ff3b5e94dec5420fba3be260e4ed8d06")

rag_instance = None

# Create OpenAI client for DeepSeek
openai_client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs
) -> str:
    """Custom LLM function that directly calls DeepSeek API without response_format"""
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add history if any
    for msg in history_messages:
        messages.append(msg)
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Call DeepSeek API directly - NO response_format parameter!
        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

async def init_rag():
    global rag_instance
    logger.info(f"Initializing RAG from {WORKING_DIR}...")
    
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
    
    embedding_func = EmbeddingFunc(
        embedding_dim=1024, # Match the dim used during indexing
        max_token_size=8192,
        func=lambda texts: hf_embed(texts, tokenizer, embed_model),
    )
    
    rag_instance = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_name=MODEL_NAME,
        embedding_func=embedding_func,
        llm_model_kwargs={}  # Empty - we handle everything in llm_model_func
    )
    await rag_instance.initialize_storages()
    await initialize_pipeline_status()
    logger.info("RAG initialization complete.")

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
        response = rag_instance.query(request.query, param=param)
        
        while asyncio.iscoroutine(response):
            response = await response
            
        return {"answer": str(response)}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    try:
        return {
            "working_dir": WORKING_DIR,
            "model": MODEL_NAME,
            "embedding_model": EMBED_MODEL_NAME,
            "status": "online"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/graph-stats")
async def get_graph_stats():    
    async def find_real_count(vdb_instance, name):
        if not vdb_instance:
            return 0
            
        try:
            # 第一步：获取内部存储对象
            # 1. 先找 client_storage
            target = getattr(vdb_instance, "client_storage", None)
            
            # 2. 如果是异步的，必须 await (这是之前报错的核心原因)
            if asyncio.iscoroutine(target) or hasattr(target, '__await__'):
                target = await target
                
            # 3. 如果没找到 client_storage，试试 _client
            if target is None:
                target = getattr(vdb_instance, "_client", None)

            # 4. 如果还是空的，甚至试试 _data
            if target is None:
                target = getattr(vdb_instance, "_data", None)

            # 第二步：暴力拆解 target 里的内容
            # 如果 target 本身就是列表，直接返回
            if isinstance(target, list):
                logger.info(f"{name} 本身就是列表，长度: {len(target)}")
                return len(target)

            # 如果 target 是字典 (就是那个长度为3的家伙)
            # 我们遍历它所有的 Value，寻找那个最长的列表
            if hasattr(target, "__dict__"):
                target = target.__dict__ # 把对象转成字典
            
            if isinstance(target, dict):
                # 打印一下所有的 Key，让你心里有数
                keys = list(target.keys())
                logger.info(f"🔍 {name} 内部包含这些 Key: {keys}")
                
                max_len = 0
                
                # 挨个检查字典里的每一个东西
                for k, v in target.items():
                    # 如果这东西是列表
                    if isinstance(v, list):
                        curr_len = len(v)
                        logger.info(f"检查 Key ['{k}']: 是列表，长度 {curr_len}")
                        if curr_len > max_len:
                            max_len = curr_len
                    # 如果这东西是另一个对象，甚至可能有 _data
                    elif hasattr(v, "_data") and isinstance(v._data, list):
                         curr_len = len(v._data)
                         logger.info(f"检查 Key ['{k}']._data: 是列表，长度 {curr_len}")
                         if curr_len > max_len:
                            max_len = curr_len
                            
                if max_len > 0:
                    logger.info(f"锁定 {name} 真实长度: {max_len}")
                    return max_len

        except Exception as e:
            logger.error(f"分析 {name} 时出错: {e}")
        
        return 0

    # 开始执行扫描
    entity_count = await find_real_count(getattr(rag_instance, 'entities_vdb', None), "Entities")
    relation_count = await find_real_count(getattr(rag_instance, 'relationships_vdb', None), "Relations")
    chunk_count = await find_real_count(getattr(rag_instance, 'chunks_vdb', None), "Chunks")
    
    return {
        "entities": entity_count,
        "relations": relation_count,
        "chunks": chunk_count,
        "corpus": "Medical",
        "model": MODEL_NAME,
        "embedding_model": EMBED_MODEL_NAME,
        "llm_base_url": LLM_BASE_URL
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)