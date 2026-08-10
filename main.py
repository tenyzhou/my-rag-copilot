import os
from openai import OpenAI

# 1. 初始化客户端（DeepSeek 接口兼容 OpenAI）
client = OpenAI(
    api_key="sk-3b32024fff8d47f08142528c8f5fdbd3",  # ⚠️ 替换为你自己的 Key
    base_url="https://api.deepseek.com"
)

print("=== DeepSeek AI 命令行助手已启动（输入 exit 退出）===")

# 2. 交互循环
while True:
    user_input = input("\n你: ")
    if user_input.strip().lower() == "exit":
        print("助手已退出！")
        break

    if not user_input.strip():
        continue

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业且有耐心的 AI 开发助手。"},
            {"role": "user", "content": user_input}
        ]
    )

    print(f"\nAI: {response.choices[0].message.content}")