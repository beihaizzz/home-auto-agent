import json
from typing import Any, Dict, List, Type, Union, Optional, TypeVar, Generic
from pydantic import BaseModel, create_model, Field
from langchain_core.documents import Document


class DeviceModelFactory:
    #TODO: 需要优化缓存，避免每次都要重复生成
    def __init__(self):
        self.type_mapping = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "number": float,
            "object": dict,
            "array": list,
        }
        # 用于登记已经动态生成的设备配置信息类型
        self.registry: Dict[str, Type[BaseModel]] = {}
        self._cache = {}

    def generate_model_from_config(self, config: Document) -> Type[BaseModel]:
        config: dict = json.loads(config.page_content)
        device_type = config["device_type"]
        # 提取参数作为属性
        properties = config["params"]["properties"]
        fields = {}

        for key, value in properties.items():
            # 根据type mapping将json字段映射为python的字段
            field_type = self.type_mapping.get(value.get("type", "string"), Any)
            # ...表示该字段是必填的
            fields[key] = (field_type, ...)
        # 动态创建pydantic类的类名
        model_name = f"{device_type.replace(' ', '')}Config"
        print(f"model_name:{model_name}")
        # 创建模型（BaseModel）
        # model = create_model(model_name, **fields)
        if model_name not in self._cache:
            # 注册，并建立映射关系
            model = create_model(model_name, **fields)
            self.registry[model_name] = model
            return model

    def generate_all(self, configs: List[Document]):
        # 批量完成模型的生成
        for config in configs:
            self.generate_model_from_config(config)

    def get_union_type(self) -> Type[BaseModel]:
        # 将所有设备模型联合起来生成一个联合类型
        # 这个可以用在 LLM 输出 schema 上，代表可能是多个模型中的任意一个。
        return Union[tuple(self.registry.values())]

    def get_model_by_type(self, device_type: str) -> Type[BaseModel]:
        # 通过模型类别名获取模型
        return self.registry.get(device_type)


ConfigT = TypeVar('ConfigT')


class Device(BaseModel, Generic[ConfigT]):
    id: str = Field(
        description="Id of the devices will be used in this scene"
    )
    type: str = Field(
        description="Type of the devices will be used in this scene,such as ac and led light"
    )
    config: ConfigT = Field(
        description="The corresponding config in device_configs for the corresponding device_id",
        default=None
    )


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class DeviceCall(BaseModel, Generic[ConfigT]):
    device_name: str
    device_id: str
    config: ConfigT = Field(
        description="the params for device_call which comes from the device_configs",
        json_schema_extra={"additionalProperties": False}  # 明确禁止额外属性
    )
    order: int = Field(
        description="The order of the device call in the scene.",
    )


class DeviceCalls(BaseModel, Generic[ConfigT]):
    device_calls: List[DeviceCall[ConfigT]] = Field(
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
