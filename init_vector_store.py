# init_vector_store.py 或在 Python/Jupyter 中执行
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
import json, os

# 初始化 Embeddings（需要 OPENAI_API_KEY）
#embeddings = OpenAIEmbeddings()
embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-08cbaae3f5b5482cac1ff6b09c7b17d9",
    model="text-embedding-v3"  # 千问官方向量模型
)

# 创建向量数据库
vector_store = Chroma(
    collection_name="vector_collection_for_agent",
    embedding_function=embeddings,
    persist_directory=os.path.join(os.getcwd(), "common", "VectorStore"),
    collection_metadata={"vs_name": "test"}
)

# 加载设备配置
with open("oneNetConfig.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [
    Document(page_content=json.dumps(device, indent=2, ensure_ascii=False))
    for device in data
]

ids = vector_store.add_documents(documents)
print(f"成功导入 {len(ids)} 个设备配置")