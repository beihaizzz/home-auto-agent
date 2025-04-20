from typing import List, Generic
from pydantic import BaseModel, Field

from common.structs import Device, DeviceCall, ConfigT


class Scene(BaseModel, Generic[ConfigT]):
    # 预设生成的情景，场景
    name: str = Field(
        description="Name of the Scene",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this Scene.",
    )
    research: bool = Field(
        description="Whether to perform web research for this Scene of the report."
    )
    start_time: str = Field(
        description="When the scene might happened,use:HH:MM:SS"
    )
    end_time: str = Field(
        description="When the scene might ended,use:HH:MM:SS"
    )
    involved_devices: List[Device[ConfigT]] = Field(
        description="Devices involved in this scene"
    )


class Scenes(BaseModel, Generic[ConfigT]):
    scenes: List[Scene] = Field(
        description="Scene of the  scheme today",
        json_schema_extra={"additionalProperties": False}  # 明确禁止额外属性
    )


class Scheme(BaseModel,Generic[ConfigT]):
    device_calls: List[DeviceCall[ConfigT]]
    scene: Scene


class Schemes(BaseModel,Generic[ConfigT]):
    schemes: List[Scheme[ConfigT]]
