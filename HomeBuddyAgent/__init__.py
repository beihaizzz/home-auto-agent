from dotenv import load_dotenv
import os

env = os.getenv("ENV", "dev")  # 默认测试环境
load_dotenv(f"../.env.dev.dev", override=True)
print(f"环境变量加载+++++++++++++++++++{os.getenv("DEEPSEEK_API_KEY")}")