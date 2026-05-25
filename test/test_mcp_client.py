"""
MCP Client 测试脚本

在没有真实服务器时，使用Mock Server测试MCP客户端功能。
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.mcp import SyncMCPClient, MCPDeviceControl


def test_device_discovery():
    """测试设备发现"""
    print("\n" + "=" * 60)
    print("测试 1: 设备发现")
    print("=" * 60)

    try:
        client = SyncMCPClient(
            server_url="http://localhost:8080",
            timeout=10
        )

        response = client.discover_devices()

        print(f"发现 {response.total_count} 个设备:")
        for device in response.devices:
            print(f"  - {device.device_name} ({device.device_type})")
            print(f"    状态: {device.status}")
            print(f"    功能: {device.capabilities}")

        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


def test_control_device():
    """测试设备控制"""
    print("\n" + "=" * 60)
    print("测试 2: 设备控制")
    print("=" * 60)

    try:
        client = SyncMCPClient(
            server_url="http://localhost:8080",
            timeout=10
        )

        # 使用字符串字面量而不是枚举
        control = MCPDeviceControl(
            device_id="dev_001",
            device_name="客厅空调",
            device_type="air_conditioner",
            action="turn_on"
        )

        print(f"发送控制指令: 打开 {control.device_name}")
        response = client.call_tool([control])

        if hasattr(response, 'overall_status'):
            print(f"执行状态: {response.overall_status}")
            for result in response.results:
                print(f"  - {result.device_name}: {result.message}")
                print(f"    设备状态: {result.device_status}")
                print(f"    执行时间: {result.execution_time:.3f}s")
        else:
            print(f"错误: {response.error_message}")

        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


def test_set_value():
    """测试设置设备值"""
    print("\n" + "=" * 60)
    print("测试 3: 设置设备值")
    print("=" * 60)

    try:
        client = SyncMCPClient(
            server_url="http://localhost:8080",
            timeout=10
        )

        # 使用字符串字面量而不是枚举
        control = MCPDeviceControl(
            device_id="dev_001",
            device_name="客厅空调",
            device_type="air_conditioner",
            action="set_value",
            parameters={"key": "temperature", "value": 24}
        )

        print(f"发送控制指令: 设置 {control.device_name} 温度为 24度")
        response = client.call_tool([control])

        if hasattr(response, 'overall_status'):
            print(f"执行状态: {response.overall_status}")
            for result in response.results:
                print(f"  - {result.device_name}: {result.message}")
                print(f"    设备状态: {result.device_status}")
        else:
            print(f"错误: {response.error_message}")

        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("MCP Client 测试套件")
    print("=" * 60)

    tests = [
        ("设备发现", test_device_discovery),
        ("设备控制", test_control_device),
        ("设置值", test_set_value)
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print("\n测试完成")


if __name__ == "__main__":
    main()