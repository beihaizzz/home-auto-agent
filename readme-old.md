# 配置信息
创建一个名为 .env.dev 的文件作为开发环境的配置文件，并添加以下内容：

```dotenv
OPENAI_API_KEY=xxx
ANTHROPIC_API_KEY=xxx
TAVILY_API_KEY=xxx
GROQ_API_KEY=xxx

DEEPSEEK_API_KEY=xxx
DEEPSEEK_API_BASE=xxx# 使用deepseek的供应商，如：https://api.siliconflow.cn/v1 硅基流动的

REDIS_PASSWORD=xxx 
```

# 下载依赖，环境初始化
切换到项目根目录下，执行以下命令安装依赖：
```shell
pip install -r requirements.txt
```

下载Langgraph cli dev
```shell
pip install -U "langgraph-cli[inmem]" #注意 ：此命令仅用于本地开发和测试，不建议用于生产环境。由于它不使用 Docker，建议使用虚拟环境来管理项目的依赖关系。
```

# 请在jupyter中运行
```jupyterpython
# 初始化模拟的设备配置信息
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
# 初始化 OpenAI Embeddings 或其他嵌入函数
embeddings = OpenAIEmbeddings()
BASE_DIR = os.getcwd()
print(BASE_DIR)
persist_directory = rf"{BASE_DIR}\common\VectorStore"
print(persist_directory)
# 创建 Chroma 对象
vector_store = Chroma(
    collection_name="vector_collection_for_agent",
    embedding_function=embeddings,
    persist_directory=persist_directory,
    collection_metadata={
        "vs_name":"test"
    }
)
```
将配置信息载入到向量数据库
```jupyterpython
from langchain_core.documents import Document
import json
BASE_DIR = os.getcwd()
# 文件路径
file_path = rf"{BASE_DIR}\oneNetConfig.json"



# 打开并读取 JSON 文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
data_documents:list[Document]=[]
# 遍历每个设备的 JSON 对象并打印
for device in data:
    dict_str=json.dumps(device, indent=2, ensure_ascii=False)
    
    # print(type(dict_str))
    data_documents.append(Document(page_content=dict_str))
ids=vector_store.add_documents(data_documents)
print(ids)
```

检查
```jupyterpython
from langchain_core.documents import Document
all=vector_store.get(
    # ids=['688875e8-b0d9-457f-ac82-68a78a4a06b4', '6803132b-29fb-46d3-9b03-7fde31994fc1', '3571348a-0d14-4231-abfa-a9684299c9d0', '4e812df8-bdd6-4371-8fed-664f231732ad', 'c7425539-357a-42b0-a6a5-0ad8bdfddccd', '1275019f-cbd4-476a-b453-df3a3b4570a3', '8415ad06-9114-4405-a222-5cc09a302451', '2bdf1514-9a6c-4663-a9f0-24a763c5cb78', '8b87e417-222e-43d9-a2f8-065e23005cfe']
)
# print(all)
for i in all["documents"]:
    doc=Document(page_content=i)
    print("="*8)
    print(doc)
```