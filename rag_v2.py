import os
import jieba
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# 1. 加载本地 .env 环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("❌ 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

# 2. 自定义 RRF (Reciprocal Rank Fusion) 混合检索器
class SimpleEnsembleRetriever:
    def __init__(self, retrievers, weights=None, c=60):
        self.retrievers = retrievers
        self.weights = weights or [0.5, 0.5]
        self.c = c

    def invoke(self, query: str):
        doc_scores = {}
        doc_map = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                doc_id = doc.page_content
                doc_map[doc_id] = doc
                score = weight * (1.0 / (rank + 1 + self.c))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:2]]


# 3. 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 4. 文档构建 BM25 与 Vector 检索器
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

def chinese_tokenizer(text: str):
    return list(jieba.cut(text))

bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_tokenizer)
bm25_retriever.k = 2

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

ensemble_retriever = SimpleEnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# 5. 主程序入口保护
if __name__ == "__main__":
    print("\n=== 🚀 RAG V2 (BM25 + 向量混合检索) 系统已就绪！(输入 exit 退出) ===")
    while True:
        user_query = input("\n请针对知识库提问: ")
        if user_query.strip().lower() == "exit":
            print("系统已退出。")
            break

        if not user_query.strip():
            continue

        retrieved_docs = ensemble_retriever.invoke(user_query)
        context_text = "\n\n".join([f"[切片{i+1}]: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        print(f"\n🔍 【混合检索召回的相关切片】:\n{context_text}")

        prompt = f"""你是一个严谨的私有知识库助手。请根据下方提供的【参考资料】，回答【用户提问】。

【原则】：
1. 优先提取资料中的具体数字、时间、规则直接回答。
2. 如果【参考资料】中完全没有相关信息，才回答“知识库中未找到相关内容”。

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

        print(f"\nAI (基于混合检索回复): {response.choices[0].message.content}")