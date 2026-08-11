import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 加载本地 .env 环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("❌ 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

# 2. 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 3. 文档加载、切分与 Chroma 向量化
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. 主程序入口保护
if __name__ == "__main__":
    print("\n=== 🚀 RAG V1 (基础向量检索系统) 已就绪！(输入 exit 退出) ===")
    while True:
        user_query = input("\n请针对知识库提问: ")
        if user_query.strip().lower() == "exit":
            print("系统已退出。")
            break

        if not user_query.strip():
            continue

        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""你是一个严谨的私有知识库助手。请根据下方提供的【参考资料】，回答【用户提问】。

【参考资料】:
{context_text}

【用户提问】:
{user_query}
"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的私有知识库问答助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        print(f"\nAI (基于向量检索回复): {response.choices[0].message.content}")