import os
import jieba
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


# -------------------------------------------------------------
# 自定义 RRF (Reciprocal Rank Fusion) 混合检索器
# 彻底脱离 LangChain 版本变动导包报错，且方便面试直观讲解算法！
# -------------------------------------------------------------
class SimpleEnsembleRetriever:
    def __init__(self, retrievers, weights=None, c=60):
        self.retrievers = retrievers
        self.weights = weights or [0.5, 0.5]
        self.c = c  # RRF 常数，工业界标准为 60

    def invoke(self, query: str):
        doc_scores = {}
        doc_map = {}

        # 遍历每一个检索器 (BM25 和 向量检索)
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                # 以文本内容作为唯一标识
                doc_id = doc.page_content
                doc_map[doc_id] = doc

                # RRF 核心打分公式：Score = weight * (1 / (rank + c))
                score = weight * (1.0 / (rank + 1 + self.c))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        # 按 RRF 得分从高到低排序，返回前 2 个最高分的切片
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:2]]


# -------------------------------------------------------------
# 1. 初始化 DeepSeek 客户端
# -------------------------------------------------------------
client = OpenAI(
    api_key="sk-3b32024fff8d47f08142528c8f5fdbd3",  # ⚠️ 替换为你申请到的真实 Key
    base_url="https://api.deepseek.com"
)

# -------------------------------------------------------------
# 2. 文档加载与切分
# -------------------------------------------------------------
print("1. 正在加载本地知识库 knowledge.txt ...")
loader = TextLoader("knowledge.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# -------------------------------------------------------------
# 3. 构建稀疏检索器 (BM25 关键词检索)
# -------------------------------------------------------------
print("2. 正在构建 BM25 稀疏检索器 (关键词匹配) ...")


def chinese_tokenizer(text: str):
    return list(jieba.cut(text))


bm25_retriever = BM25Retriever.from_documents(
    chunks,
    preprocess_func=chinese_tokenizer
)
bm25_retriever.k = 2

# -------------------------------------------------------------
# 4. 构建密集检索器 (Chroma 向量检索)
# -------------------------------------------------------------
print("3. 正在构建 Chroma 密集检索器 (向量语义匹配) ...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -------------------------------------------------------------
# 5. 融合构建混合检索器 (Hybrid Retriever)
# -------------------------------------------------------------
print("4. 正在融合构建 Hybrid 混合检索器 (自定义 RRF 算法) ...")
ensemble_retriever = SimpleEnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

print("\n=== 🚀 RAG V2 (BM25 + 向量混合检索) 系统已就绪！(输入 exit 退出) ===")

# -------------------------------------------------------------
# 6. 混合检索问答循环
# -------------------------------------------------------------
while True:
    user_query = input("\n请针对知识库提问: ")
    if user_query.strip().lower() == "exit":
        print("问答系统已退出。")
        break

    if not user_query.strip():
        continue

    # 执行混合检索
    retrieved_docs = ensemble_retriever.invoke(user_query)

    # 打印检索到的文本切片
    context_text = "\n\n".join([f"[切片{i + 1}]: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
    print(f"\n🔍 【混合检索召回的相关切片】:\n{context_text}")

    # 构造增强 Prompt
    prompt = f"""你是一个严谨的私有知识库助手。请根据下方提供的【参考资料】，回答【用户提问】。

【原则】：
1. 优先提取资料中的具体数字、时间、规则直接回答。
2. 如果【参考资料】中完全没有相关信息，才回答“知识库中未找到相关内容”。

【参考资料】:
{context_text}

【用户问题】:
{user_query}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严谨的私有知识库问答助手。"},
            {"role": "user", "content": prompt}
        ]
    )

    print(f"\nAI (结合混合检索回复): {response.choices[0].message.content}")