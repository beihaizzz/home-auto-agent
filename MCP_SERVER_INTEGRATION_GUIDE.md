# MCP Server 对接指南

本文档详细说明外部设备控制 Server 如何与 Home Auto Agent 项目对接。

## 概述

Home Auto Agent 作为 **MCP Client**，通过标准 Model Context Protocol (MCP) 与外部设备控制 Server 通信。

**对接架构**：

```
Home Auto Agent (MCP Client)
        │
        ▼ HTTP POST /mcp/v1/control
        │
    Your MCP Server
        │
        ▼
    设备控制执行
```

## 核心接口

### 1. 设备控制接口

**Endpoint**: `POST /mcp/v1/control`

**HTTP Method**: POST

**Content-Type**: `application/json`

#### 请求格式

```json
{
"version": "1.0",
"payload_type": "request",
"request_id": "uuid-string",
"timestamp": "2024-01-01T00:00:00Z",
"source": "home-auto-agent",
"session_id": "optional-session-id",
"user_id": "optional-user-id",
"controls": [
    {
    "device_id": "设备唯一标识",
    "device_name": "设备名称",
    "device_type": "设备类型",
    "action": "动作类型",
    "parameters": {"参数名": "参数值"}
    }
],
"timeout": 30,
"execute_strategy": "sequential"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| version | string | 是 | 协议版本，固定为 "1.0" |
| payload_type | string | 是 | 负载类型，固定为 "request" |
| request_id | string | 是 | 请求唯一标识（UUID） |
| timestamp | string | 是 | 请求时间戳（ISO 8601格式） |
| source | string | 是 | 请求来源，通常为 "home-auto-agent" |
| session_id | string | 否 | 会话ID |
| user_id | string | 否 | 用户ID |
| controls | array | 是 | 设备控制指令列表，至少包含一个 |
| timeout | int | 否 | 超时时间，默认30秒 |
| execute_strategy | string | 否 | 执行策略："sequential"（顺序）或 "parallel"（并行） |

#### 响应格式

```json
{
"version": "1.0",
"payload_type": "response",
"request_id": "uuid-string",
"timestamp": "2024-01-01T00:00:00Z",
"overall_status": "success",
"results": [
    {
    "device_id": "设备唯一标识",
    "device_name": "设备名称",
    "status": "success",
    "message": "执行结果消息",
    "action": "执行的动作",
    "parameters": {"参数名": "参数值"},
    "device_status": {"设备状态键": "状态值"},
    "execution_time": 0.5
    }
],
"session_id": "optional-session-id",
"metadata": {}
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| version | string | 是 | 协议版本，固定为 "1.0" |
| payload_type | string | 是 | 负载类型，固定为 "response" |
| request_id | string | 是 | 对应请求的 request_id |
| timestamp | string | 是 | 响应时间戳 |
| overall_status | string | 是 | 整体状态：success/failed/pending/timeout/invalid_param |
| results | array | 是 | 每个设备控制的执行结果 |
| session_id | string | 否 | 会话ID |
| metadata | object | 否 | 附加元数据 |

#### 错误响应格式

```json
{
"version": "1.0",
"payload_type": "error",
"request_id": "uuid-string",
"timestamp": "2024-01-01T00:00:00Z",
"error_code": "错误码",
"error_message": "错误消息",
"error_details": {"详细信息": "..."}
}
```

## 支持的设备类型

| 设备类型 | 说明 | 支持的动作 |
|---------|------|-----------|
| `air_conditioner` | 空调 | turn_on, turn_off, set_value, get_status, toggle |
| `light` | 灯光 | turn_on, turn_off, set_value, get_status, toggle |
| `curtain` | 窗帘 | turn_on, turn_off, set_value, get_status, toggle |
| `heater` | 加热器 | turn_on, turn_off, set_value, get_status, toggle |
| `fan` | 风扇 | turn_on, turn_off, set_value, get_status, toggle |
| `tv` | 电视 | turn_on, turn_off, set_value, get_status, toggle |
| `speaker` | 音箱 | turn_on, turn_off, set_value, get_status, toggle |
| `switch` | 开关 | turn_on, turn_off, set_value, get_status, toggle |
| `sensor` | 传感器 | get_status |
| `scene` | 场景 | scene_activate |
| `humidifier` | 加湿器 | turn_on, turn_off, set_value, get_status, toggle |

## 支持的动作类型

| 动作 | 说明 | 是否需要参数 |
|------|------|-------------|
| `turn_on` | 打开设备 | 否 |
| `turn_off` | 关闭设备 | 否 |
| `set_value` | 设置参数值 | 是（至少一个参数） |
| `get_status` | 获取设备状态 | 否 |
| `toggle` | 切换设备状态 | 否 |
| `scene_activate` | 激活场景 | 否 |

## 设备参数规范

### 空调 (air_conditioner)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| power | boolean | true/false | 开关状态 |
| temperature | integer | 16-30 | 设定温度(℃) |
| fan_speed | integer | 1-5 | 风速级别 |
| mode | string | 冷/热/除湿/自动 | 运行模式 |

### 灯光 (light)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| power | boolean | true/false | 开关状态 |
| brightness | integer | 0-100 | 亮度百分比 |
| color | string | RGB值 | 颜色 |

### 风扇 (fan)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| power | boolean | true/false | 开关状态 |
| fan_speed | integer | 1-5 | 风速级别 |
| oscillation | boolean | true/false | 摇头功能 |

### 加湿器 (humidifier)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| power | boolean | true/false | 开关状态 |
| humidity_level | integer | 30-80 | 湿度百分比 |
| mist_output | integer | 1-3 | 雾量级别 |

### 电视 (tv)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| power | boolean | true/false | 开关状态 |
| volume | integer | 0-100 | 音量 |
| channel | integer | 1-999 | 频道号 |
| picture_mode | string | 标准/动态/电影/自定义 | 画质模式 |

### 音箱 (speaker)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| power | boolean | true/false | 开关状态 |
| volume | integer | 0-100 | 音量 |
| sound_mode | string | 音乐/电影/游戏/新闻 | 音效模式 |

### 窗帘 (curtain)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| power | boolean | true/false | 开关状态 |
| position | integer | 0-100 | 位置百分比 |
| auto_mode | boolean | true/false | 自动模式 |

### 门锁 (lock)

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|---------|------|
| locked | boolean | true/false | 锁定状态 |
| fingerprint_unlock | boolean | true/false | 指纹开锁 |
| alarm | boolean | true/false | 防盗警报 |

## 实现示例

### Python FastAPI 示例

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

app = FastAPI(title="MCP Device Control Server")

# 设备状态存储（模拟）
device_states = {
"aX23Jrf5xy": {"power": "off", "temperature": 26, "fan_speed": 1},
"mD97Vya1hi": {"power": "off"},
"pR65Xtq3lc": {"power": "off", "fan_speed": 1},
"zD81Vyr8jh": {"power": "off", "humidity_level": 40}
}

# 数据模型
class MCPDeviceControl(BaseModel):
    device_id: str
    device_name: str
    device_type: str
    action: str
    parameters: Optional[Dict[str, Any]] = None

class MCPRequest(BaseModel):
    version: str = "1.0"
    payload_type: str = "request"
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str
    controls: List[MCPDeviceControl]
    timeout: int = 30
    execute_strategy: str = "sequential"

class MCPDeviceResult(BaseModel):
    device_id: str
    device_name: str
    status: str
    message: str
    action: str
    parameters: Optional[Dict[str, Any]] = None
    device_status: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

class MCPResponse(BaseModel):
    version: str = "1.0"
    payload_type: str = "response"
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    overall_status: str
    results: List[MCPDeviceResult]

@app.post("/mcp/v1/control", response_model=MCPResponse)
async def control_devices(request: MCPRequest):
    """设备控制接口"""
    results = []
    
    for control in request.controls:
        device_id = control.device_id
        device_name = control.device_name
        action = control.action
        parameters = control.parameters or {}
        
        # 检查设备是否存在
        if device_id not in device_states:
            results.append(MCPDeviceResult(
                device_id=device_id,
                device_name=device_name,
                status="failed",
                message=f"设备 {device_id} 不存在",
                action=action
            ))
            continue
        
        # 执行动作
        try:
            device_status = device_states[device_id]
            
            if action == "turn_on":
                device_status["power"] = "on"
                message = f"{device_name} 已打开"
                
            elif action == "turn_off":
                device_status["power"] = "off"
                message = f"{device_name} 已关闭"
                
            elif action == "set_value":
                # 设置参数
                for key, value in parameters.items():
                    if key in device_status:
                        device_status[key] = value
                message = f"{device_name} 参数已更新: {parameters}"
                
            elif action == "get_status":
                message = f"获取 {device_name} 状态成功"
                
            elif action == "toggle":
                current_power = device_status.get("power", "off")
                device_status["power"] = "off" if current_power == "on" else "on"
                message = f"{device_name} 状态已切换"
                
            else:
                results.append(MCPDeviceResult(
                    device_id=device_id,
                    device_name=device_name,
                    status="failed",
                    message=f"不支持的动作: {action}",
                    action=action
                ))
                continue
                
            results.append(MCPDeviceResult(
                device_id=device_id,
                device_name=device_name,
                status="success",
                message=message,
                action=action,
                parameters=parameters,
                device_status=device_status.copy()
            ))
            
        except Exception as e:
            results.append(MCPDeviceResult(
                device_id=device_id,
                device_name=device_name,
                status="failed",
                message=f"执行失败: {str(e)}",
                action=action
            ))
    
    # 判断整体状态
    overall_status = "success" if all(r.status == "success" for r in results) else "failed"
    
    return MCPResponse(
        request_id=request.request_id,
        overall_status=overall_status,
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## 测试验证

### 使用 curl 测试

```bash
curl -X POST http://localhost:8080/mcp/v1/control \
-H "Content-Type: application/json" \
-d '{
"version": "1.0",
"payload_type": "request",
"request_id": "test-request-001",
"timestamp": "2024-01-01T00:00:00Z",
"source": "home-auto-agent",
"controls": [
    {
    "device_id": "aX23Jrf5xy",
    "device_name": "空调",
    "device_type": "air_conditioner",
    "action": "set_value",
    "parameters": {"temperature": 24}
    }
],
"timeout": 30,
"execute_strategy": "sequential"
}'
```

### 预期响应

```json
{
"version": "1.0",
"payload_type": "response",
"request_id": "test-request-001",
"timestamp": "2024-01-01T00:00:00Z",
"overall_status": "success",
"results": [
    {
    "device_id": "aX23Jrf5xy",
    "device_name": "空调",
    "status": "success",
    "message": "空调 参数已更新: {'temperature': 24}",
    "action": "set_value",
    "parameters": {"temperature": 24},
    "device_status": {"power": "off", "temperature": 24, "fan_speed": 1}
    }
]
}
```

## 部署建议

### 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| MCP_SERVER_URL | Server 地址 | http://localhost:8080 |
| MCP_API_KEY | API 密钥（可选） | 空 |
| MCP_TIMEOUT | 请求超时时间 | 30 |

### 安全建议

1. **认证机制**：建议使用 API Key 认证，在请求头中传递 `Authorization: Bearer <api-key>`
2. **HTTPS**：生产环境建议启用 HTTPS
3. **请求验证**：验证请求格式和参数合法性
4. **日志记录**：记录所有请求和响应，便于排查问题

## 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 422 错误 | 请求格式不正确 | 检查 JSON 格式和必填字段 |
| 设备不存在 | device_id 错误 | 确认设备 ID 是否正确 |
| 动作不支持 | action 字段值错误 | 检查动作类型是否在支持列表中 |
| 连接失败 | Server 未启动或地址错误 | 确认 Server 运行状态和 MCP_SERVER_URL 配置 |

## 版本兼容性

| MCP 版本 | 状态 |
|----------|------|
| 1.0 | 当前支持 |
