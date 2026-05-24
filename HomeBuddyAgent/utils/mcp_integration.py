"""
MCP Integration Module for HomeBuddyAgent

This module handles the integration between HomeBuddyAgent and external MCP servers.
It converts device control commands to MCP format and communicates with MCP Server
using the standard call_tool method defined in the MCP protocol specification.

设备映射自动加载机制：
1. 从 oneNetConfig.json 初始化向量库时，自动生成 device_mappings.json
2. 本模块启动时自动加载 device_mappings.json
3. 支持动态更新映射配置
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from common.mcp import (
    SyncMCPClient, MCPDeviceControl,
    MCPResponse, MCPError
)
from common.structs import DeviceCall, DeviceCalls, DeviceResult
from HomeBuddyAgent.utils.state import State
from common.configuration import Configuration
from langchain_core.runnables import RunnableConfig


class MCPIntegrationError(Exception):
    """MCP集成异常"""
    pass


class MCPDeviceController:
    """
    MCP设备控制器
    
    使用官方标准的MCP call_tool方法与外部MCP Server通信。
    支持从 device_mappings.json 自动加载设备映射配置。
    """
    
    def __init__(self):
        self._client = None
        self._server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
        self._api_key = os.getenv("MCP_API_KEY")
        
        # 设备映射配置
        self._device_type_mappings: Dict[str, Dict] = {}
        self._action_mappings: Dict[str, List[str]] = {}
        self._reverse_device_mapping: Dict[str, str] = {}  # 别名 → MCP类型
        self._reverse_action_mapping: Dict[str, str] = {}  # 别名 → 标准动作
        
        # 自动加载映射配置
        self._load_mappings()
    
    def _load_mappings(self):
        """从配置文件加载设备映射"""
        mapping_file = self._get_mapping_file_path()
        
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    mappings = json.load(f)
                
                # 加载设备类型映射
                device_mappings = mappings.get("device_type_mappings", {})
                for device_type, info in device_mappings.items():
                    self._device_type_mappings[device_type] = info
                    
                    # 构建反向映射
                    mcp_type = info.get("mcp_type", device_type)
                    for alias in info.get("aliases", []):
                        self._reverse_device_mapping[alias] = mcp_type
                    self._reverse_device_mapping[device_type] = mcp_type
                
                # 加载动作映射
                action_mappings = mappings.get("action_mappings", {})
                for action, aliases in action_mappings.items():
                    self._action_mappings[action] = aliases
                    for alias in aliases:
                        self._reverse_action_mapping[alias] = action
                    self._reverse_action_mapping[action] = action
                
                print(f"[MCP] 已加载设备映射配置:")
                print(f"  - 设备类型: {len(self._device_type_mappings)} 个")
                print(f"  - 动作类型: {len(self._action_mappings)} 个")
                print(f"  - 来源: {mapping_file}")
                
            except Exception as e:
                print(f"[MCP] 加载映射配置失败: {e}")
                self._load_default_mappings()
        else:
            print(f"[MCP] 映射配置文件不存在: {mapping_file}")
            print(f"[MCP] 请先运行: python init_vector_store.py")
            self._load_default_mappings()
    
    def _get_mapping_file_path(self) -> str:
        """获取映射配置文件路径"""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "common", "mcp", "device_mappings.json"
        )
    
    def _load_default_mappings(self):
        """加载默认映射（兼容旧版本）"""
        print("[MCP] 使用默认映射配置")
        
        default_device_mappings = {
            "air_conditioner": {"mcp_type": "air_conditioner", "aliases": ["空调", "Air Conditioner"]},
            "light": {"mcp_type": "light", "aliases": ["灯", "LED Light", "灯光"]},
            "fan": {"mcp_type": "fan", "aliases": ["风扇", "Electric Fan"]},
            "curtain": {"mcp_type": "curtain", "aliases": ["窗帘", "Smart Curtain"]},
            "tv": {"mcp_type": "tv", "aliases": ["电视", "Smart TV"]},
            "speaker": {"mcp_type": "speaker", "aliases": ["音箱", "Speaker"]},
            "heater": {"mcp_type": "heater", "aliases": ["加热器", "Heater"]},
            "humidifier": {"mcp_type": "humidifier", "aliases": ["加湿器", "Humidifier"]},
            "lock": {"mcp_type": "lock", "aliases": ["门锁", "Smart Lock"]},
            "switch": {"mcp_type": "switch", "aliases": ["开关", "switch"]},
        }
        
        for device_type, info in default_device_mappings.items():
            self._device_type_mappings[device_type] = info
            for alias in info["aliases"]:
                self._reverse_device_mapping[alias] = info["mcp_type"]
        
        default_action_mappings = {
            "turn_on": ["打开", "开启", "启动", "开"],
            "turn_off": ["关闭", "关掉", "停止", "关"],
            "set_value": ["设置", "设定", "调节"],
            "get_status": ["状态", "查询", "查看"],
            "toggle": ["切换"],
        }
        
        for action, aliases in default_action_mappings.items():
            self._action_mappings[action] = aliases
            for alias in aliases:
                self._reverse_action_mapping[alias] = action
    
    def reload_mappings(self):
        """重新加载映射配置"""
        print("[MCP] 重新加载映射配置...")
        self._device_type_mappings.clear()
        self._action_mappings.clear()
        self._reverse_device_mapping.clear()
        self._reverse_action_mapping.clear()
        self._load_mappings()
    
    def _map_device_type(self, device_type: str) -> str:
        """
        映射设备类型到MCP标准类型
        
        Args:
            device_type: 设备类型字符串
            
        Returns:
            MCP标准类型
        """
        if not device_type:
            return "switch"
        
        # 1. 直接匹配
        if device_type in self._reverse_device_mapping:
            return self._reverse_device_mapping[device_type]
        
        # 2. 模糊匹配（检查是否包含关键词）
        device_type_lower = device_type.lower()
        for alias, mcp_type in self._reverse_device_mapping.items():
            if alias.lower() in device_type_lower or device_type_lower in alias.lower():
                return mcp_type
        
        # 3. 未知类型返回原值（不丢失信息）
        print(f"[MCP] 未知设备类型: {device_type}，直接传递原值")
        return device_type
    
    def _map_action_type(self, action: str) -> str:
        """
        映射动作类型到MCP标准动作
        
        Args:
            action: 动作字符串
            
        Returns:
            MCP标准动作
        """
        if not action:
            return "turn_on"
        
        # 1. 直接匹配
        if action in self._reverse_action_mapping:
            return self._reverse_action_mapping[action]
        
        # 2. 模糊匹配
        action_lower = action.lower()
        for alias, standard_action in self._reverse_action_mapping.items():
            if alias.lower() in action_lower or action_lower in alias.lower():
                return standard_action
        
        # 3. 未知动作返回原值
        print(f"[MCP] 未知动作类型: {action}，直接传递原值")
        return action
    
    @property
    def client(self) -> SyncMCPClient:
        """获取或创建MCP客户端"""
        if self._client is None:
            self._client = SyncMCPClient(
                server_url=self._server_url,
                api_key=self._api_key,
                timeout=30,
                max_retries=3
            )
        return self._client
    
    def convert_to_mcp_control(self, device_call: DeviceCall) -> MCPDeviceControl:
        """
        将设备调用转换为MCP设备控制指令
        
        Args:
            device_call: 设备调用对象
            
        Returns:
            MCPDeviceControl: MCP设备控制指令
        """
        action_str = device_call.action
        parameters = device_call.parameters or {}
        
        # 映射动作和设备类型
        mcp_action = self._map_action_type(action_str)
        device_type_str = parameters.get("device_type", "switch")
        mcp_device_type = self._map_device_type(device_type_str)
        
        # 处理参数
        mcp_parameters: Dict[str, Any] = {}
        
        if mcp_action == "set_value" or action_str in ["设置", "设定", "调节"]:
            # 提取设置值
            for key, value in parameters.items():
                if key not in ["device_type", "device_name", "action"]:
                    mcp_parameters["key"] = key
                    mcp_parameters["value"] = value
                    break
        
        return MCPDeviceControl(
            device_id=device_call.device_id,
            device_name=device_call.device_name,
            device_type=mcp_device_type,
            action=mcp_action,
            parameters=mcp_parameters if mcp_parameters else None
        )
    
    def convert_to_mcp_controls(self, device_calls: DeviceCalls) -> List[MCPDeviceControl]:
        """将设备调用列表转换为MCP控制指令列表"""
        controls = []
        for call in device_calls.device_calls:
            control = self.convert_to_mcp_control(call)
            controls.append(control)
        return controls
    
    def execute_mcp_request(
        self,
        device_calls: DeviceCalls,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> MCPResponse:
        """执行MCP设备控制请求"""
        controls = self.convert_to_mcp_controls(device_calls)
        
        if not controls:
            raise MCPIntegrationError("没有有效的设备控制指令")
        
        try:
            response = self.client.call_tool(
                controls=controls,
                session_id=session_id,
                user_id=user_id,
                execute_strategy="sequential"
            )
            return response
        except Exception as e:
            raise MCPIntegrationError(f"MCP请求失败: {str(e)}")
    
    def convert_mcp_response_to_device_results(
        self,
        mcp_response: Union[MCPResponse, MCPError]
    ) -> List[DeviceResult]:
        """将MCP响应转换为设备结果列表"""
        results = []
        
        if isinstance(mcp_response, MCPError):
            results.append(DeviceResult(
                success=False,
                message=f"MCP错误: {mcp_response.error_message}",
                device_id="",
                device_name="",
                data={"error_code": mcp_response.error_code}
            ))
            return results
        
        for device_result in mcp_response.results:
            success = device_result.status == "success"
            
            result = DeviceResult(
                success=success,
                message=device_result.message,
                device_id=device_result.device_id,
                device_name=device_result.device_name,
                data=device_result.device_status or {}
            )
            results.append(result)
        
        return results


# 全局单例
mcp_controller = MCPDeviceController()


def mcp_device_call(state: State) -> Dict[str, Any]:
    """
    MCP设备调用节点 - 使用标准call_tool方法调用外部MCP Server
    
    工作流程:
    1. 接收设备控制指令 (device_calls)
    2. 转换为MCP格式
    3. 使用call_tool方法调用外部MCP Server
    4. 接收返回结果并转换
    5. 更新状态
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    print("---MCP设备调用---")
    
    device_calls = state.get("device_calls")
    
    if not device_calls or not device_calls.device_calls:
        print("没有设备调用指令")
        return {
            "device_call_results": [],
            "feed_back": False
        }
    
    try:
        response = mcp_controller.execute_mcp_request(
            device_calls=device_calls,
            session_id=state.get("session_id"),
            user_id=state.get("user_id")
        )
        
        device_results = mcp_controller.convert_mcp_response_to_device_results(response)
        
        if isinstance(response, MCPError):
            all_success = False
        else:
            all_success = all(r.success for r in device_results)
        
        print(f"MCP调用完成，状态: {response.overall_status if hasattr(response, 'overall_status') else 'error'}")
        
        return {
            "device_call_results": device_results,
            "feed_back": all_success,
            "mcp_response": response
        }
    
    except MCPIntegrationError as e:
        print(f"MCP调用失败: {str(e)}")
        
        error_results = []
        for call in device_calls.device_calls:
            error_results.append(DeviceResult(
                success=False,
                message=f"MCP调用失败: {str(e)}",
                device_id=call.device_id,
                device_name=call.device_name,
                data={}
            ))
        
        return {
            "device_call_results": error_results,
            "feed_back": False
        }