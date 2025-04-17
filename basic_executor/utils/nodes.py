from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache

from basic_executor.utils.prompts import prompt_for_tool
from basic_executor.utils.state import State
from basic_executor.utils.tools import tools
from langgraph.prebuilt import ToolNode

from common.common_utils import get_model
from common.configuration import Configuration

OPENAI_KEY = ("***REMOVED***"
              "-***REMOVED***")




@lru_cache(maxsize=4)
def _rag_loder():
    # with open("D:\\DevelopFiles\\pycharms\\Stimulate\\AI\\files\\function_file.json", encoding="UTF-8") as f:
    #     file = f.read()
    #
    # text_splitter = CharacterTextSplitter(
    #     separator="\n\n",
    #     chunk_size=5,
    #     chunk_overlap=2,
    #     length_function=len,
    #     is_separator_regex=False
    # )
    #
    # texts = text_splitter.create_documents([file])
    # vector_store = Chroma.from_documents(texts, OpenAIEmbeddings())
    persist_directory = r"D:\DevelopFiles\pycharms\Command_parser_langgraph\my_agent\ChromaDB\test"

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_KEY
    )

    vector_store = Chroma(
        collection_name="vector_collection_for_agent",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def retrieve(state: State):
    # print("retrieve")
    vector_store = _rag_loder()
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs, "tool_using": False}


def generate(state: State,config: RunnableConfig):
    print("generate")
    configurable = Configuration.from_runnable_config(config)
    model = get_model(
        model_provider=configurable.tool_call_provider,
        model_name=configurable.tool_call_model
    ).bind_tools(tools)
    docs_content = "\n\n".join(doc.page_content for doc in state["device_configs"])
    messages = state["messages"]
    if not state["tool_using"]:
        messages = messages + prompt_for_tool.invoke(
            {"question": [state["question"]], "context": [docs_content]}).to_messages()
        # print("messages",messages)
    response = model.invoke(messages)
    # llm.with_structured_output(
    #
    # )
    # print(response)
    return {
        "answer": response.content,
        "device_configs": [],
        "messages": messages + [response]
    }


tool_node = ToolNode(tools)
