from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal, Optional


class AdditionalInfo(BaseModel):
    type: Literal["tavily_search_results_json", "last_similar_device_call", "suggested_device_call"] \
        = Field(description="The type of source for additional information is usually the same as the source of the "
                            "information, that is, the name of the tool being called")
    # 目前只写了web search，其他两个类型还没写
    content: str | None = Field(default=None, description="Content of the information.")
    url: str | None = Field(default=None,
                            description="URL of the searched information from tavily_search_results_json(web "
                                        "search).if web search were never used,it is none ")


class RouterScore(BaseModel):
    """用于路由检测"""
    command_score: Literal["agent", "executor"] = Field(description="用户指令的评分")
    info_for_agent: None | str = Field(description="若是路由到agent的，则为agent为完成设备操作要做的事情和获取的信息，"
                                                   "根据用户的指令而来。如果不是路由到agent则为空", default=None)


class ClarityScore(BaseModel):
    """用于指令清晰度检查的评分"""
    clarity_score: str = Field(description="Clarity score 'clear' or 'ambiguous'", default="")
    missing_info: str = Field(description="缺失的信息描述，如果没有则为空字符串", default="")


class AdditionalInfos(BaseModel):
    search_infos: List[AdditionalInfo]
