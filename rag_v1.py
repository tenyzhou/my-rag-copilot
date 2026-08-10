import os
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 初始化 DeepSeek 客户端
client = OpenAI(
    api_key="sk-3b32024fff8d47f08142528c8f5fdbd3",  # ⚠️ 替换为你申请到的真实 Key
    base_url="https://api.deepseek.com"
)

# 2. 加载文档 -> 文本切片 -> 存入向量数据库 Chroma
print("1. 正在加载本地知识库 knowledge.txt ...")
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()

print("2. 正在对文档进行文本切分 (Chunking) ...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

print("3. 正在计算向量并构建向量数据库 ...")
# 使用免费的轻量级中文 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("\n=== 🚀 RAG 知识库问答系统 (V1 版本) 已就绪！(输入 exit 退出) ===")

# 3. 检索 + 结合上下文问答循环
while True:
    user_query = input("\n请针对知识库提问: ")
    if user_query.strip().lower() == "exit":
        print("问答系统已退出。")
        break

    if not user_query.strip():
        continue

    # 检索最相关文档片段
    retrieved_docs = retriever.invoke(user_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 构造增强提示词 Prompt
    prompt = f"""请严格根据下方参考资料回答用户的问题。如果资料中未提及，请直接回答“知识库中未找到相关内容”。

【参考资料】:
{context_text}

【用户问题】:
{user_query}
"""

    # 调用 DeepSeek 生成回答
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严谨的私有知识库问答助手。"},
            {"role": "user", "content": prompt}
        ]
    )

    print(f"\nAI (基于知识库回复): {response.choices[0].message.content}")