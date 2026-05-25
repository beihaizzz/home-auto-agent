# init_vector_store.py 或在 Python/Jupyter 中执行
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
import json, os
from datetime import datetime
from dotenv import load_dotenv
import argparse

# 加载.env.dev文件
load_dotenv('.env.dev', override=True)

# 解析命令行参数
parser = argparse.ArgumentParser(description="初始化向量数据库")
parser.add_argument('--provider', type=str, default='qwen', choices=['openai', 'qwen', 'anthropic', 'deepseek', 'groq'],
                    help='embedding模型提供商')
args = parser.parse_args()

# 根据提供商选择embedding模型
if args.provider == "openai":
    embeddings = OpenAIEmbeddings()
elif args.provider == "qwen":
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v3"  # 千问官方向量模型
    )
elif args.provider == "anthropic":
    # Anthropic没有官方的embedding模型，使用OpenAI作为替代
    embeddings = OpenAIEmbeddings()
elif args.provider == "deepseek":
    # DeepSeek的embedding模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE')
    )
elif args.provider == "groq":
    # Groq没有官方的embedding模型，使用OpenAI作为替代
    embeddings = OpenAIEmbeddings()
else:
    # 默认使用OpenAI的embedding模型
    embeddings = OpenAIEmbeddings()

print(f"使用 {args.provider} 的embedding模型")

# 创建向量数据库
vector_store = Chroma(
    collection_name="vector_collection_for_agent",
    embedding_function=embeddings,
    persist_directory=os.path.join(os.getcwd(), "common", "VectorStore"),
    collection_metadata={"vs_name": "test"}
)

# 加载设备配置
with open("oneNetConfig.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [
    Document(page_content=json.dumps(device, indent=2, ensure_ascii=False))
    for device in data
]

ids = vector_store.add_documents(documents)
print(f"成功导入 {len(ids)} 个设备配置")

# ========== 新增：自动生成设备映射配置 ==========

def generate_device_mappings(config_file: str = "oneNetConfig.json", 
                            output_file: str = "common/mcp/device_mappings.json"):
    """
    从设备配置文件自动生成MCP设备映射配置
    
    Args:
        config_file: 设备配置文件路径
        output_file: 映射配置文件输出路径
    """
    print("\n" + "="*60)
    print("生成设备映射配置")
    print("="*60)
    
    # 读取设备配置
    with open(config_file, "r", encoding="utf-8") as f:
        devices = json.load(f)
    
    device_type_mappings = {}
    action_mappings = {
        # 默认动作映射
        "turn_on": ["打开", "开启", "启动", "开", "turn_on", "on"],
        "turn_off": ["关闭", "关掉", "停止", "关", "turn_off", "off"],
        "set_value": ["设置", "设定", "调节", "调整", "set_value"],
        "get_status": ["状态", "查询", "查看", "get_status"],
        "toggle": ["切换", "toggle"],
    }
    
    # 从设备配置中提取设备类型和参数
    for device in devices:
        device_type = device.get("device_type", "")
        device_name = device.get("device_name", {}).get("value", "")
        params = device.get("params", {}).get("properties", {})
        
        if not device_type:
            continue
        
        # 添加设备类型映射
        if device_type not in device_type_mappings:
            # 根据设备类型猜测MCP标准类型
            mcp_type = _guess_mcp_type(device_type)
            
            device_type_mappings[device_type] = {
                "mcp_type": mcp_type,
                "aliases": [device_type, device_name],
                "params": params
            }
        else:
            # 添加设备名称到别名列表
            if device_name and device_name not in device_type_mappings[device_type]["aliases"]:
                device_type_mappings[device_type]["aliases"].append(device_name)
        
        # 从参数中提取动作
        for param_name, param_info in params.items():
            if param_name not in action_mappings:
                action_mappings[param_name] = [param_name]
                # 添加中文描述
                description = param_info.get("description", "")
                if description and description not in action_mappings[param_name]:
                    action_mappings[param_name].append(description)
    
    # 构建映射配置
    mappings = {
        "device_type_mappings": device_type_mappings,
        "action_mappings": action_mappings,
        "last_updated": datetime.now().isoformat(),
        "source_file": config_file,
        "total_devices": len(devices),
        "total_device_types": len(device_type_mappings)
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入映射配置
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 设备类型映射: {len(device_type_mappings)} 个")
    for dt, info in device_type_mappings.items():
        print(f"   - {dt} → {info['mcp_type']}")
    
    print(f"✅ 动作类型映射: {len(action_mappings)} 个")
    print(f"✅ 映射配置已保存: {output_file}")
    
    return mappings


def _guess_mcp_type(device_type: str) -> str:
    """
    根据设备类型猜测MCP标准类型
    
    Args:
        device_type: 设备类型字符串
        
    Returns:
        MCP标准类型
    """
    # 关键词映射表
    type_keywords = {
        "light": ["light", "led", "灯", "灯光", "照明"],
        "air_conditioner": ["air conditioner", "空调", "冷气", "ac"],
        "fan": ["fan", "风扇", "电扇"],
        "curtain": ["curtain", "窗帘", "卷帘", "遮阳"],
        "tv": ["tv", "television", "电视", "电视机"],
        "speaker": ["speaker", "音箱", "音响", "音频"],
        "heater": ["heater", "加热器", "暖气", "取暖"],
        "humidifier": ["humidifier", "加湿器", "加湿"],
        "lock": ["lock", "门锁", "智能锁", "指纹锁"],
        "switch": ["switch", "开关", "插座", "battery", "电池"],
        "sensor": ["sensor", "传感器", "感应器"],
        "vacuum": ["vacuum", "扫地", "清洁机器人", "扫地机"]
    }
    
    device_type_lower = device_type.lower().replace("_", " ").replace("-", " ")
    
    for mcp_type, keywords in type_keywords.items():
        for keyword in keywords:
            if keyword in device_type_lower:
                return mcp_type
    
    # 无法识别，返回处理后的原类型
    return device_type_lower.replace(" ", "_")


if __name__ == "__main__":
    # 原有的向量库初始化代码
    # ... (保持不变)
    
    # 新增：生成设备映射配置
    generate_device_mappings()
    
    print("\n" + "="*60)
    print("初始化完成！")
    print("="*60)