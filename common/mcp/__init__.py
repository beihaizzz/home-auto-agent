"""
Model Context Protocol (MCP) Module

This module provides the implementation of the Model Context Protocol Client,
which enables communication between AI agents and external device control systems.

Key components:
- protocol.py: MCP message format definitions (shared with server)
- client.py: MCP client implementation for sending requests to external server

Official MCP Protocol Specification:
- Supports standard call_tool method for device control
- Defines standardized request/response formats
- Ensures interoperability with any MCP-compliant server
"""

from .protocol import (
    MCPVersion,
    MCPPayloadType,
    MCPActionType,
    MCPDeviceType,
    MCPStatus,
    MCPDeviceControl,
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPDeviceDiscoveryRequest,
    MCPDeviceDiscoveryResponse,
    MCPDeviceInfo,
    MCPDeviceResult
)

from .client import (
    MCPClient,
    SyncMCPClient,
    MCPClientError,
    MCPClientConnectionError,
    MCPClientTimeoutError,
    MCPClientValidationError
)

__version__ = "1.0.0"
__author__ = "Home Auto Agent Team"

__all__ = [
    # Protocol classes - shared with MCP Server
    "MCPVersion",
    "MCPPayloadType",
    "MCPActionType",
    "MCPDeviceType",
    "MCPStatus",
    "MCPDeviceControl",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "MCPDeviceDiscoveryRequest",
    "MCPDeviceDiscoveryResponse",
    "MCPDeviceInfo",
    "MCPDeviceResult",
    
    # Client classes - used to call external MCP Server
    "MCPClient",
    "SyncMCPClient",
    "MCPClientError",
    "MCPClientConnectionError",
    "MCPClientTimeoutError",
    "MCPClientValidationError",
]