# from typing import Annotated
#
# import requests
# from langchain_core.messages import ToolMessage
# from langchain_core.tools import tool, InjectedToolCallId
# from langgraph.types import Command
#
# from common.structs import DeviceCall,ConfigT
#
#
# def _process_device_call(device_commands: list[DeviceCall[ConfigT]]):
#     """
#     根据设备操作名称和参数处理设备调用。
#     """
#     for command in device_commands:
#         print(command.device_name)
#         print(command.params)
#     return "操作成功"
#
#
# @tool
# def device_call(device_commands: list[DeviceCall[ConfigT]], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     提供模型操作设备
#     """
#     return Command(
#         update={
#             "tool_using": True,
#             "messages": [
#                 ToolMessage(
#                     content=_process_device_call(device_commands), tool_call_id=tool_call_id
#                 )
#             ]
#         }
#     )
#
#
# @tool
# def get_weather(province: str = "四川", city: str = "成都", county: str = "郫都区") -> str:
#     """获取成都市的实时天气信息。"""
#     base_url = 'https://wis.qq.com/weather/common'
#     params = {
#         'source': 'pc',
#         'weather_type': 'observe',
#         'province': province,
#         'city': city,
#         'county': county
#     }
#
#     response = requests.get(base_url, params=params)
#
#     if response.status_code == 200:
#         data = response.json()
#         if data['status'] == 200:
#             observe = data['data']['observe']
#             weather = observe['weather']
#             temperature = observe['degree']
#             humidity = observe['humidity']
#             wind_direction = observe['wind_direction_name']
#             wind_power = observe['wind_power']
#             return (
#                 f"成都市 {county} 的天气是 {weather}，温度是 {temperature}°C，"
#                 f"湿度为 {humidity}%，风向为 {wind_direction}，风力为 {wind_power}。"
#             )
#         else:
#             return "无法获取成都市的天气信息，返回的数据状态异常。"
#     else:
#         return "无法连接天气服务，请检查网络连接。"
#
#
# tools = [get_weather, device_call]
