from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_for_tool_str = """
你是一个智慧家庭助手，根据我的指令操作不同的智能家居(使用工具)，得到工具结果后给出反馈
"""

prompt_for_tool = ChatPromptTemplate(
    [
        (SystemMessage(content=prompt_for_tool_str)),
        (MessagesPlaceholder("context", )),
        (MessagesPlaceholder("question"))
    ]
)
