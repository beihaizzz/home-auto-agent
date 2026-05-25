"""
MCP Mock Server - 用于测试的模拟MCP服务器

这个模块提供了一个简单的MCP服务器实现，用于在没有真实服务时测试MCP客户端功能。
"""

import os
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.mcp.protocol import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPDeviceControl,
    MCPDeviceDiscoveryResponse,
    MCPDeviceInfo,
    MCPDeviceResult,
    MCPVersion
)

# 模拟设备数据
MOCK_DEVICES = {
    "mD97Vya1hi": {
        "device_id": "mD97Vya1hi",
        "device_name": "LED灯",
        "device_type": "light",
        "location": "卧室",
        "status": {"power": "off"},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    },
    "aX23Jrf5xy": {
        "device_id": "aX23Jrf5xy",
        "device_name": "空调",
        "device_type": "air_conditioner",
        "location": "客厅",
        "status": {"power": "off", "temperature": 26},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    },
    "pR65Xtq3lc": {
        "device_id": "pR65Xtq3lc",
        "device_name": "风扇",
        "device_type": "fan",
        "location": "客厅",
        "status": {"power": "off", "fan_speed": 1, "oscillation": False},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    },
    "zD81Vyr8jh": {
        "device_id": "zD81Vyr8jh",
        "device_name": "加湿器",
        "device_type": "humidifier",
        "location": "卧室",
        "status": {"power": "off", "humidity_level": 40, "mist_output": 2},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    },
    "vC38Ltz9mn": {
        "device_id": "vC38Ltz9mn",
        "device_name": "音箱",
        "device_type": "speaker",
        "location": "客厅",
        "status": {"power": "off", "volume": 50, "sound_mode": "音乐"},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    },
    "xE74Jql1pv": {
        "device_id": "xE74Jql1pv",
        "device_name": "电池",
        "device_type": "battery",
        "location": "卧室",
        "status": {"battery_display": True, "charging": False, "power_saving": False},
        "capabilities": ["set_value", "get_status"]
    },
    "cD89Tgh2xz": {
        "device_id": "cD89Tgh2xz",
        "device_name": "电视",
        "device_type": "tv",
        "location": "客厅",
        "status": {"power": "off", "volume": 30, "channel": 1, "picture_mode": "标准"},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    },
    "wJ12Xtm6yb": {
        "device_id": "wJ12Xtm6yb",
        "device_name": "窗帘",
        "device_type": "curtain",
        "location": "卧室",
        "status": {"power": "off", "position": 0, "auto_mode": False},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    },
    "mN41Kyq9zf": {
        "device_id": "mN41Kyq9zf",
        "device_name": "门锁",
        "device_type": "lock",
        "location": "大门",
        "status": {"locked": True, "fingerprint_unlock": True, "alarm": False},
        "capabilities": ["turn_on", "turn_off", "set_value", "get_status"]
    }
}

# 设备状态存储
device_states = {
    "mD97Vya1hi": {"power": "off"},
    "aX23Jrf5xy": {"power": "off", "temperature": 26, "fan_speed": 1, "mode": "自动"},
    "pR65Xtq3lc": {"power": "off", "fan_speed": 1, "oscillation": False},
    "zD81Vyr8jh": {"power": "off", "humidity_level": 40, "mist_output": 2},
    "vC38Ltz9mn": {"power": "off", "volume": 50, "sound_mode": "音乐"},
    "xE74Jql1pv": {"battery_display": True, "charging": False, "power_saving": False},
    "cD89Tgh2xz": {"power": "off", "volume": 30, "channel": 1, "picture_mode": "标准"},
    "wJ12Xtm6yb": {"power": "off", "position": 0, "auto_mode": False},
    "mN41Kyq9zf": {"locked": True, "fingerprint_unlock": True, "alarm": False}
}

# 初始化FastAPI应用
app = FastAPI(
    title="MCP Mock Server",
    description="Model Context Protocol Mock Server - for testing purposes",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "version": MCPVersion.CURRENT,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/mcp/v1/control")
async def handle_control_request(request: MCPRequest):
    """
    处理设备控制请求

    这是MCP协议的核心接口，接收设备控制指令并返回执行结果。
    """
    print(f"[MCP Mock Server] 收到控制请求: {request.request_id}")
    print(f"[MCP Mock Server] 控制指令: {[c.device_name for c in request.controls]}")

    # 验证协议版本
    if request.version != MCPVersion.CURRENT:
        return MCPError(
            request_id=request.request_id,
            error_code="UNSUPPORTED_VERSION",
            error_message=f"不支持的协议版本: {request.version}"
        )

    try:
        results: List[MCPDeviceResult] = []

        # 执行每个控制指令
        for control in request.controls:
            device_result = await execute_device_control(control)
            results.append(device_result)

        # 确定整体状态
        overall_status = "success"
        if any(r.status == "failed" for r in results):
            overall_status = "failed"

        # 构建响应
        response = MCPResponse(
            request_id=request.request_id,
            overall_status=overall_status,
            results=results,
            session_id=request.session_id
        )

        print(f"[MCP Mock Server] 执行完成，整体状态: {overall_status}")
        return response

    except Exception as e:
        print(f"[MCP Mock Server] 执行失败: {str(e)}")
        return MCPError(
            request_id=request.request_id,
            error_code="INTERNAL_ERROR",
            error_message=f"服务器内部错误: {str(e)}"
        )


async def execute_device_control(control: MCPDeviceControl) -> MCPDeviceResult:
    """
    执行单个设备控制指令
    """
    start_time = datetime.now()

    try:
        # 检查设备是否存在
        if control.device_id not in MOCK_DEVICES:
            return MCPDeviceResult(
                device_id=control.device_id,
                device_name=control.device_name,
                status="failed",
                message=f"设备 {control.device_id} 不存在",
                action=control.action,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        # 执行动作
        device_info = MOCK_DEVICES[control.device_id]
        success = True
        message = ""
        action = control.action

        if action == "turn_on":
            device_states[control.device_id]["power"] = "on"
            message = f"已打开 {device_info['device_name']}"
            
            # 处理附加参数（如 temperature, fan_speed 等）
            params = control.parameters or {}
            additional = params.get("additional", {})
            if additional:
                for key, value in additional.items():
                    device_states[control.device_id][key] = value
                    message += f", {key}设为{value}"
            
            # 处理批量设置参数
            batch = params.get("batch", {})
            if batch:
                for key, value in batch.items():
                    device_states[control.device_id][key] = value
                    message += f", {key}设为{value}"
        elif action == "turn_off":
            device_states[control.device_id]["power"] = "off"
            message = f"已关闭 {device_info['device_name']}"
        elif action == "set_value":
            params = control.parameters or {}
            
            # 处理批量设置参数
            batch = params.get("batch", {})
            if batch:
                for key, value in batch.items():
                    device_states[control.device_id][key] = value
                    message += f"{device_info['device_name']} 的 {key} 已设置为 {value}; "
                message = message.rstrip("; ")
            else:
                # 单参数设置
                key = params.get("key", "temperature")
                value = params.get("value")

                if value is not None:
                    device_states[control.device_id][key] = value
                    message = f"{device_info['device_name']} 的 {key} 已设置为 {value}"
                else:
                    success = False
                    message = "缺少value参数"
        elif action == "get_status":
            message = f"获取 {device_info['device_name']} 状态成功"
        elif action == "toggle":
            current_power = device_states[control.device_id].get("power", "off")
            new_power = "on" if current_power == "off" else "off"
            device_states[control.device_id]["power"] = new_power
            message = f"{device_info['device_name']} 已切换为 {new_power}"
        else:
            success = False
            message = f"不支持的动作: {action}"

        # 构建结果
        return MCPDeviceResult(
            device_id=control.device_id,
            device_name=control.device_name,
            status="success" if success else "failed",
            message=message,
            action=action,
            parameters=control.parameters,
            device_status=device_states[control.device_id].copy(),
            execution_time=(datetime.now() - start_time).total_seconds()
        )

    except Exception as e:
        return MCPDeviceResult(
            device_id=control.device_id,
            device_name=control.device_name,
            status="failed",
            message=f"执行失败: {str(e)}",
            action=control.action,
            execution_time=(datetime.now() - start_time).total_seconds()
        )


@app.get("/mcp/v1/devices")
async def discover_devices(
    device_type: Optional[str] = None,
    location: Optional[str] = None
):
    """
    发现设备

    返回所有可用的设备，支持按类型和位置过滤。
    """
    print(f"[MCP Mock Server] 收到设备发现请求: type={device_type}, location={location}")

    try:
        devices = []
        for device_data in MOCK_DEVICES.values():
            # 类型过滤
            if device_type and device_data["device_type"] != device_type:
                continue
            # 位置过滤
            if location and device_data["location"] != location:
                continue

            # 合并当前状态
            device_with_status = device_data.copy()
            device_with_status["status"] = device_states[device_data["device_id"]].copy()
            devices.append(MCPDeviceInfo(**device_with_status))

        response = MCPDeviceDiscoveryResponse(
            request_id="discover_" + str(datetime.now().timestamp()),
            devices=devices,
            total_count=len(devices)
        )

        print(f"[MCP Mock Server] 发现 {len(devices)} 个设备")
        return response

    except Exception as e:
        return MCPError(
            request_id="discover_" + str(datetime.now().timestamp()),
            error_code="INTERNAL_ERROR",
            error_message=f"获取设备列表失败: {str(e)}"
        )


@app.get("/mcp/v1/devices/{device_id}")
async def get_device_info(device_id: str):
    """
    获取单个设备信息
    """
    print(f"[MCP Mock Server] 收到获取设备信息请求: {device_id}")

    if device_id not in MOCK_DEVICES:
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 不存在")

    device_data = MOCK_DEVICES[device_id].copy()
    device_data["status"] = device_states[device_id].copy()
    return MCPDeviceInfo(**device_data)


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("MCP Mock Server - 正在启动")
    print("=" * 60)
    print(f"服务器地址: http://0.0.0.0:8080")
    print(f"MCP控制接口: http://0.0.0.0:8080/mcp/v1/control")
    print(f"MCP设备发现: http://0.0.0.0:8080/mcp/v1/devices")
    print("=" * 60)
    print("\n已注册的模拟设备:")
    for device_id, device_data in MOCK_DEVICES.items():
        print(f"  - {device_data['device_name']} ({device_data['device_type']})")
    print("\n按 Ctrl+C 停止服务器\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)