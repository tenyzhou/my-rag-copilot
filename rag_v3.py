import os
import jieba
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIError
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# -------------------------------------------------------------
# 1. 加载环境变量与 API Key (配置网络重试与超时)
# -------------------------------------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("❌ 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

# 🎯 增加 max_retries=3 和 timeout=30.0，大幅提升抗网络抖动能力
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    max_retries=3,
    timeout=30.0
)


# -------------------------------------------------------------
# 2. 多格式通用文档加载器
# -------------------------------------------------------------
def load_all_documents(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []

    documents = []

    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents.extend(txt_loader.load())

    md_loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents.extend(md_loader.load())

    try:
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"⚠️ PDF 加载提示: {e}")

    print(f"📚 成功加载了 {len(documents)} 个文档页面！")
    return documents


# -------------------------------------------------------------
# 3. 文档切分与索引建立
# -------------------------------------------------------------
print("1. 正在加载 data/ 目录下的知识库文件 ...")
documents = load_all_documents("data")

if not documents:
    raise ValueError("❌ data/ 目录下没有任何有效文档，请放入至少一个 .txt, .md 或 .pdf 文件！")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# -------------------------------------------------------------
# 4. 构建第一阶段：BM25 + 向量 混合召回器
# -------------------------------------------------------------
print("2. 正在构建 BM25 稀疏检索器与 Chroma 密集向量检索器 ...")


def chinese_tokenizer(text: str):
    return list(jieba.cut(text))


bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_tokenizer)
bm25_retriever.k = 6

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# -------------------------------------------------------------
# 5. 构建第二阶段：Reranker 重排序器
# -------------------------------------------------------------
print("3. 正在加载 BGE Reranker 重排序模型 (BAAI/bge-reranker-base) ...")
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def rewrite_query(original_query: str) -> str:
    prompt = f"""你是一个搜索改写专家。请将用户口语化的提问，改写为1句利于文档检索的标准搜索词。只输出改写后的句子。

【用户原始提问】：{original_query}
【改写后标准问题】："""

    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        rewritten = res.choices[0].message.content.strip()
        print(f"🔄 [Query Rewriter 改写]: '{original_query}' ──> '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"⚠️ 改写跳过 (网络抖动): {e}")
        return original_query


def two_stage_rag_search(query: str, top_n_recall=6, top_k_rerank=3):
    search_q = rewrite_query(query)

    bm25_docs = bm25_retriever.invoke(search_q)
    vector_docs = vector_retriever.invoke(search_q)

    candidate_docs = []
    seen = set()
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            candidate_docs.append(doc)

    pair_inputs = [[search_q, doc.page_content] for doc in candidate_docs]
    scores = reranker_model.predict(pair_inputs)

    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    final_docs = [doc for doc, score in scored_docs[:top_k_rerank]]
    return final_docs


# -------------------------------------------------------------
# 6. 主程序运行问答循环 (含异常网络捕获)
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== 🚀 RAG V3 (支持逻辑推理 + 抗网络抖动版) 已就绪！(输入 exit 退出) ===")
    while True:
        user_query = input("\n请针对知识库提问: ")
        if user_query.strip().lower() == "exit":
            print("系统已退出。")
            break

        if not user_query.strip():
            continue

        final_retrieved_docs = two_stage_rag_search(user_query, top_n_recall=6, top_k_rerank=3)
        context_text = "\n\n".join([f"[切片{i + 1}]: {doc.page_content}" for i, doc in enumerate(final_retrieved_docs)])

        print(f"\n🔍 【精排选出的相关切片】:\n{context_text}")

        prompt = f"""你是一个严谨且具备逻辑推理能力的私有知识库助手。请根据下方提供的【参考资料】，回答【用户提问】。

【回答原则与逻辑推理要求】：
1. **直接提取**：优先从【参考资料】中明确提及的具体数字、时间、规则直接提取并回答。
2. **合理推导**：若资料中未直接给出显式答案（例如未直接写下班时间），但包含充分的关联信息（如写明了“9:30 上班”或“每日工作8小时”），你可以进行常理推算，并在回答中**明确说明推导过程**。
3. **严格拒答**：若资料中完全没有任何关联事实依据（如健身房设施完全未提及），请说明该部分内容未找到。

【参考资料】:
{context_text}

【用户提问】:
{user_query}
"""

        # 🎯 捕获网络异常，防止程序直接中断崩溃
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个严谨的私有知识库问答助手。"},
                    {"role": "user", "content": prompt}
                ]
            )
            print(f"\nAI (基于检索与逻辑推导回复):\n{response.choices[0].message.content}")
        except (APIConnectionError, APIError) as e:
            print(f"\n❌ 网络请求超时或 SSL 握手失败: {e}")
            print("💡 建议：请检查梯子/代理设置（可尝试关闭或开启直连），然后重新输入问题提问！")
        except Exception as e:
            print(f"\n❌ 未知异常: {e}")