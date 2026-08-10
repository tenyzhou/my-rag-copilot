import os
import jieba
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

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
# 3. 构造阶段一：BM25 + 向量 混合召回 (Recall 阶段，扩大召回量到 Top 6)
# -------------------------------------------------------------
print("2. 构建第一阶段：混合召回器 (BM25 + Chroma Vector) ...")


def chinese_tokenizer(text: str):
    return list(jieba.cut(text))


bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_tokenizer)
bm25_retriever.k = 4

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------------------
# 4. 构造阶段二：Cross-Encoder 重排序器 (Rerank 阶段)
# -------------------------------------------------------------
print("3. 正在加载 BGE Reranker 重排序模型 (BAAI/bge-reranker-base) ...")
# CrossEncoder 模型会将 (Query, Document) 作为一个整体输入计算全局 Attention 分数
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def two_stage_rag_search(query: str, top_n_recall=6, top_k_rerank=2):
    # 第一阶段：初筛召回 (Recall)
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    # 召回结果去重
    candidate_docs = []
    seen_contents = set()
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            candidate_docs.append(doc)

    print(f"\n🔍 【阶段一召回切片数量】: {len(candidate_docs)} 个")

    # 第二阶段：重排序 (Rerank)
    # 构造 (Query, Document) 对送入 Cross-Encoder
    pair_inputs = [[query, doc.page_content] for doc in candidate_docs]
    scores = reranker_model.predict(pair_inputs)

    # 将得分与文档绑定并降序排列
    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 【阶段二 Rerank 重排序得分榜】:")
    for rank, (doc, score) in enumerate(scored_docs):
        print(f"   - Rank {rank + 1} [得分: {score:.4f}]: {doc.page_content[:40]}...")

    # 截取最终精排后的 Top-K 切片
    final_docs = [doc for doc, score in scored_docs[:top_k_rerank]]
    return final_docs


print("\n=== 🚀 RAG V3 (两阶段检索：双路召回 + Rerank 重排序) 已就绪！(输入 exit 退出) ===")

# -------------------------------------------------------------
# 5. RAG V3 问答交互循环
# -------------------------------------------------------------
while True:
    user_query = input("\n请针对知识库提问: ")
    if user_query.strip().lower() == "exit":
        print("问答系统已退出。")
        break

    if not user_query.strip():
        continue

    # 执行两阶段检索
    final_retrieved_docs = two_stage_rag_search(user_query, top_n_recall=6, top_k_rerank=2)
    context_text = "\n\n".join([f"[精选切片{i + 1}]: {doc.page_content}" for i, doc in enumerate(final_retrieved_docs)])

    # 构造增强 Prompt
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

    print(f"\nAI (基于两阶段检索回复): {response.choices[0].message.content}")