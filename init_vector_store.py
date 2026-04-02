# init_vector_store.py 或在 Python/Jupyter 中执行
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
import json, os
from dotenv import load_dotenv
import argparse

# 加载.env.dev文件
load_dotenv('.env.dev', override=True)

# 解析命令行参数
parser = argparse.ArgumentParser(description="初始化向量数据库")
parser.add_argument('--provider', type=str, default='qwen', choices=['openai', 'qwen', 'anthropic', 'deepseek', 'groq'],
                    help='embedding模型提供商')
args = parser.parse_args()

# 根据提供商选择embedding模型
if args.provider == "openai":
    embeddings = OpenAIEmbeddings()
elif args.provider == "qwen":
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v3"  # 千问官方向量模型
    )
elif args.provider == "anthropic":
    # Anthropic没有官方的embedding模型，使用OpenAI作为替代
    embeddings = OpenAIEmbeddings()
elif args.provider == "deepseek":
    # DeepSeek的embedding模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE')
    )
elif args.provider == "groq":
    # Groq没有官方的embedding模型，使用OpenAI作为替代
    embeddings = OpenAIEmbeddings()
else:
    # 默认使用OpenAI的embedding模型
    embeddings = OpenAIEmbeddings()

print(f"使用 {args.provider} 的embedding模型")

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