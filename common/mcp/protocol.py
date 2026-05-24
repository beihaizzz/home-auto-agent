"""
Model Context Protocol (MCP) - Protocol Definitions

This module defines the standard MCP protocol data structures according to
the official Model Context Protocol specification.
"""

from typing import Optional, Dict, Any, List, Literal, Union
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field


class MCPVersion:
    """MCP协议版本"""
    CURRENT = "1.0"
    SUPPORTED_VERSIONS = ["1.0"]


MCPPayloadType = Literal["request", "response", "error"]
"""MCP负载类型"""

MCPActionType = Literal[
    "turn_on", "turn_off", "set_value", "get_status", "toggle", "scene_activate"
]
"""MCP动作类型"""

MCPDeviceType = Literal[
    "air_conditioner", "light", "curtain", "heater", "fan",
    "tv", "speaker", "switch", "sensor", "scene"
]
"""MCP设备类型"""

MCPStatus = Literal["success", "failed", "pending", "timeout", "invalid_param"]
"""MCP状态码"""

class MCPDeviceControl(BaseModel):
    """MCP设备控制指令"""
    device_id: str = Field(..., description="设备唯一标识")
    device_name: str = Field(..., description="设备名称")
    device_type: MCPDeviceType = Field(..., description="设备类型")
    action: MCPActionType = Field(..., description="动作类型")
    parameters: Optional[Dict[str, Any]] = Field(None, description="动作参数")


class MCPRequest(BaseModel):
    """MCP请求消息"""
    version: str = Field(MCPVersion.CURRENT, description="协议版本")
    payload_type: MCPPayloadType = Field("request", description="负载类型")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="请求唯一标识")
    timestamp: datetime = Field(default_factory=datetime.now, description="请求时间戳")
    source: str = Field(..., description="请求来源标识")
    session_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    controls: List[MCPDeviceControl] = Field(..., description="设备控制指令列表", min_items=1)
    timeout: Optional[int] = Field(30, description="超时时间(秒)")
    execute_strategy: Literal["sequential", "parallel"] = Field("sequential", description="执行策略")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")


class MCPDeviceResult(BaseModel):
    """MCP设备执行结果"""
    device_id: str = Field(..., description="设备ID")
    device_name: str = Field(..., description="设备名称")
    status: MCPStatus = Field(..., description="执行状态")
    message: str = Field(..., description="执行结果消息")
    action: MCPActionType = Field(..., description="执行的动作")
    parameters: Optional[Dict[str, Any]] = Field(None, description="执行时的参数")
    device_status: Optional[Dict[str, Any]] = Field(None, description="设备当前状态")
    execution_time: Optional[float] = Field(None, description="执行耗时(秒)")


class MCPResponse(BaseModel):
    """MCP响应消息"""
    version: str = Field(MCPVersion.CURRENT, description="协议版本")
    payload_type: MCPPayloadType = Field("response", description="负载类型")
    request_id: str = Field(..., description="对应请求ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    overall_status: MCPStatus = Field(..., description="整体执行状态")
    results: List[MCPDeviceResult] = Field(..., description="设备执行结果列表")
    session_id: Optional[str] = Field(None, description="会话ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")


class MCPError(BaseModel):
    """MCP错误响应"""
    version: str = Field(MCPVersion.CURRENT, description="协议版本")
    payload_type: MCPPayloadType = Field("error", description="负载类型")
    request_id: str = Field(..., description="对应请求ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间戳")
    error_code: str = Field(..., description="错误码")
    error_message: str = Field(..., description="错误消息")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")


class MCPDeviceInfo(BaseModel):
    """MCP设备信息"""
    device_id: str = Field(..., description="设备ID")
    device_name: str = Field(..., description="设备名称")
    device_type: MCPDeviceType = Field(..., description="设备类型")
    location: Optional[str] = Field(None, description="设备位置")
    status: Dict[str, Any] = Field(..., description="设备当前状态")
    capabilities: List[str] = Field(..., description="设备支持的动作列表")


class MCPDeviceDiscoveryRequest(BaseModel):
    """MCP设备发现请求"""
    version: str = Field(MCPVersion.CURRENT, description="协议版本")
    payload_type: MCPPayloadType = Field("request", description="负载类型")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="请求唯一标识")
    timestamp: datetime = Field(default_factory=datetime.now, description="请求时间戳")
    device_type: Optional[str] = Field(None, description="设备类型过滤")
    location: Optional[str] = Field(None, description="位置过滤")


class MCPDeviceDiscoveryResponse(BaseModel):
    """MCP设备发现响应"""
    version: str = Field(MCPVersion.CURRENT, description="协议版本")
    payload_type: MCPPayloadType = Field("response", description="负载类型")
    request_id: str = Field(..., description="对应请求ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    devices: List[MCPDeviceInfo] = Field(..., description="设备列表")
    total_count: int = Field(..., description="设备总数")