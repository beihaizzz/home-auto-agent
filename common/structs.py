from typing import List, Dict, Any
from pydantic import BaseModel, Field

class GenericConfig(BaseModel):
    key: str | None = None  # 示例字段

    class Config:
        extra = "forbid"

class Device(BaseModel):
    id: str = Field(
        description="Id of the devices will be used in this scene"
    )
    type: str = Field(
        description="Type of the devices will be used in this scene,such as ac and led light"
    )
    config: GenericConfig | None | None = Field(
        description="The corresponding config in device_configs for the corresponding device_id",
        default=None
    )



class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class DeviceCall(BaseModel):
    device_name: str
    device_id: str
    params: Dict[str, Any] = Field(
        description="the params for device_call which comes from the device_configs",
        json_schema_extra={"additionalProperties": False}  # 明确禁止额外属性
    )
    order: int = Field(
        description="The order of the device call in the scene.",
        ge=0,
    )


class DeviceCalls(BaseModel):
    device_calls: List[DeviceCall] = Field(
        description="List of device calls.",
    )


class DeviceResult(BaseModel):
    success: bool = Field(
        description="Whether the device call was successful.",
    )
    message: str = Field(
        description="Message of the device call.",
    )
    data: Any
