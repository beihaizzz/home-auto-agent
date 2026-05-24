"""
Model Context Protocol (MCP) - Client Implementation

This module implements the MCP Client according to the official specification.
It provides methods to communicate with external MCP Servers using standard
MCP protocol methods like call_tool.
"""

import os
import json
import requests
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from urllib.parse import urljoin

from .protocol import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPDeviceControl,
    MCPDeviceDiscoveryRequest,
    MCPDeviceDiscoveryResponse,
    MCPDeviceInfo,
    MCPVersion,
    MCPPayloadType,
    MCPStatus,
    MCPDeviceType,
    MCPActionType
)


class MCPClientError(Exception):
    """MCP客户端异常基类"""
    pass


class MCPClientConnectionError(MCPClientError):
    """MCP连接异常"""
    pass


class MCPClientTimeoutError(MCPClientError):
    """MCP超时异常"""
    pass


class MCPClientValidationError(MCPClientError):
    """MCP验证异常"""
    pass


class SyncMCPClient:
    """
    MCP同步客户端实现
    
    根据官方MCP协议规范，提供标准的call_tool方法与MCP Server通信。
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        source: str = "home-auto-agent"
    ):
        """
        初始化MCP客户端
        
        Args:
            server_url: MCP Server地址，默认为环境变量MCP_SERVER_URL
            api_key: API密钥，默认为环境变量MCP_API_KEY
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            source: 请求来源标识
        """
        self.server_url = server_url or os.getenv("MCP_SERVER_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("MCP_API_KEY")
        self.timeout = timeout or int(os.getenv("MCP_TIMEOUT", "30"))
        self.max_retries = max_retries or int(os.getenv("MCP_MAX_RETRIES", "3"))
        self.source = source
        self._session = requests.Session()
        
        # 设置默认请求头
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-MCP-Version": MCPVersion.CURRENT
        })
        
        if self.api_key:
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })
    
    def _generate_request_id(self) -> str:
        """生成唯一请求ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _send_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        method: str = "POST"
    ) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            endpoint: API端点
            payload: 请求体
            method: HTTP方法
        
        Returns:
            响应数据
        
        Raises:
            MCPClientConnectionError: 连接失败
            MCPClientTimeoutError: 请求超时
            MCPClientError: 其他错误
        """
        url = urljoin(self.server_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries - 1:
                    raise MCPClientConnectionError(f"无法连接到MCP Server: {url}")
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise MCPClientTimeoutError(f"MCP请求超时: {url}")
            except requests.exceptions.RequestException as e:
                raise MCPClientError(f"MCP请求失败: {str(e)}")
        
        raise MCPClientError("请求失败，已达到最大重试次数")
    
    def call_tool(
        self,
        controls: List[MCPDeviceControl],
        **kwargs
    ) -> Union[MCPResponse, MCPError]:
        """
        官方标准MCP call_tool方法
        
        这是MCP协议的核心方法，用于调用外部MCP Server的设备控制功能。
        
        Args:
            controls: 设备控制指令列表
            **kwargs: 可选参数（session_id, user_id, timeout, execute_strategy, metadata）
        
        Returns:
            MCPResponse: 成功响应
            MCPError: 错误响应
        """
        return self.send_control_request(controls, **kwargs)
    
    def send_control_request(
        self,
        controls: List[MCPDeviceControl],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: Optional[int] = None,
        execute_strategy: str = "sequential",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[MCPResponse, MCPError]:
        """
        发送设备控制请求
        
        Args:
            controls: 设备控制指令列表
            session_id: 会话ID
            user_id: 用户ID
            timeout: 超时时间
            execute_strategy: 执行策略
            metadata: 附加元数据
        
        Returns:
            MCPResponse或MCPError
        """
        # 验证输入
        if not controls or len(controls) == 0:
            raise MCPClientValidationError("设备控制指令列表不能为空")
        
        # 构建MCP请求
        request = MCPRequest(
            request_id=self._generate_request_id(),
            source=self.source,
            session_id=session_id,
            user_id=user_id,
            controls=controls,
            timeout=timeout or self.timeout,
            execute_strategy=execute_strategy,
            metadata=metadata
        )
        
        # 转换为JSON格式
        payload = json.loads(request.model_dump_json())
        
        try:
            # 调用MCP Server的控制接口
            response_data = self._send_request(
                endpoint="/mcp/v1/control",
                payload=payload
            )
            
            # 根据响应类型解析
            payload_type = response_data.get("payload_type")
            
            if payload_type == "error":
                return MCPError(**response_data)
            else:
                return MCPResponse(**response_data)
        
        except MCPClientValidationError:
            raise
        except Exception as e:
            # 返回错误响应
            return MCPError(
                request_id=request.request_id,
                error_code="CLIENT_ERROR",
                error_message=f"客户端请求失败: {str(e)}"
            )
    
    def discover_devices(
        self,
        device_type: Optional[str] = None,
        location: Optional[str] = None
    ) -> Union[MCPDeviceDiscoveryResponse, MCPError]:
        """
        发现设备
        
        Args:
            device_type: 设备类型过滤
            location: 位置过滤
        
        Returns:
            MCPDeviceDiscoveryResponse或MCPError
        """
        try:
            params = {}
            if device_type:
                params["device_type"] = device_type
            if location:
                params["location"] = location
            
            response_data = self._send_request(
                endpoint="/mcp/v1/devices",
                payload=params,
                method="GET"
            )
            
            return MCPDeviceDiscoveryResponse(**response_data)
        
        except Exception as e:
            return MCPError(
                request_id=self._generate_request_id(),
                error_code="DISCOVERY_FAILED",
                error_message=f"设备发现失败: {str(e)}"
            )
    
    def get_device_info(self, device_id: str) -> Union[MCPDeviceInfo, MCPError]:
        """
        获取设备信息
        
        Args:
            device_id: 设备ID
        
        Returns:
            MCPDeviceInfo或MCPError
        """
        try:
            response_data = self._send_request(
                endpoint=f"/mcp/v1/devices/{device_id}",
                payload={},
                method="GET"
            )
            
            return MCPDeviceInfo(**response_data)
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return MCPError(
                    request_id=self._generate_request_id(),
                    error_code="DEVICE_NOT_FOUND",
                    error_message=f"设备 {device_id} 不存在"
                )
            else:
                return MCPError(
                    request_id=self._generate_request_id(),
                    error_code="GET_DEVICE_FAILED",
                    error_message=f"获取设备信息失败: {str(e)}"
                )
        except Exception as e:
            return MCPError(
                request_id=self._generate_request_id(),
                error_code="GET_DEVICE_FAILED",
                error_message=f"获取设备信息失败: {str(e)}"
            )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()


class MCPClient(SyncMCPClient):
    """
    MCP客户端别名
    
    为保持向后兼容性，提供MCPClient作为SyncMCPClient的别名。
    """
    pass