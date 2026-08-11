import os
import jieba
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# 导入我们前两步写好的增量索引与多模态解析模块
from incremental_indexer import check_incremental_changes, save_manifest
from multimodal_parser import process_image_file

# -------------------------------------------------------------
# 1. 初始化环境变量与 DeepSeek 客户端
# -------------------------------------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("❌ 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# -------------------------------------------------------------
# 2. 融合增量检测与多模态图片解析的通用全量加载器
# -------------------------------------------------------------
def load_all_multimodal_documents(data_dir="data"):
    """
    全量文档加载器：支持 TXT / MD / PDF 文本解析 + PNG / JPG 图片 OCR+VLM 解析
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []

    # 1. 执行 MD5 增量检测
    added, modified, deleted, new_manifest = check_incremental_changes(data_dir)
    print(f"📊 [增量检测报告]: 新增 {len(added)} 个, 修改 {len(modified)} 个, 删除 {len(deleted)} 个")
    save_manifest(new_manifest)

    documents = []

    # 2. 遍历加载 TXT, MD, PDF 文本文件
    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents.extend(txt_loader.load())

    md_loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents.extend(md_loader.load())

    try:
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"⚠️ PDF 加载提示: {e}")

    # 3. 遍历加载 PNG / JPG 图片文件（调用多模态解析）
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                try:
                    img_doc = process_image_file(img_path)
                    documents.append(img_doc)
                except Exception as e:
                    print(f"⚠️ 图片 {file} 多模态解析跳过: {e}")

    print(f"📚 成功全量读入并构建了 {len(documents)} 个多模态 Document 对象！")
    return documents


# -------------------------------------------------------------
# 3. 建立多模态 Chroma 向量库与 BM25 索引
# -------------------------------------------------------------
print("1. 正在初始化知识库索引 (多模态加载 + 文本切片 + 向量化) ...")
raw_docs = load_all_multimodal_documents("data")

if not raw_docs:
    raise ValueError("❌ data/ 目录下没有任何有效文档，请放入 txt/md/pdf/png 文件！")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(raw_docs)


def chinese_tokenizer(text: str):
    return list(jieba.cut(text))


bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_tokenizer)
bm25_retriever.k = 6

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

reranker_model = CrossEncoder("BAAI/bge-reranker-base")


# -------------------------------------------------------------
# 4. 在线模块 A：问题意图路由器 (Question Router)
# -------------------------------------------------------------
def question_router(user_query: str) -> str:
    """
    轻量意图分类节点：区分是闲聊 (chitchat) 还是 领域知识库提问 (knowledge_query)
    """
    router_prompt = f"""你是一个意图分类助手。请判断用户输入的提问类型。

【类型定义】：
- chitchat: 打招呼、感谢、日常闲聊、关于AI身份的提问（如“你好”、“你是谁”、“谢谢”、“今天天气真好”）。
- knowledge_query: 关于公司制度、规章、技术架构、代码规范、电话号码、文档内容的实质性知识提问。

请仅输出类型名称（chitchat 或 knowledge_query），不要带有任何标点和多余解释。
【用户输入】：{user_query}
【分类结果】："""

    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0.0
        )
        intent = res.choices[0].message.content.strip().lower()
        return "chitchat" if "chitchat" in intent else "knowledge_query"
    except Exception:
        return "knowledge_query"


# -------------------------------------------------------------
# 5. 在线模块 B：Query 改写 (Query Rewriter)
# -------------------------------------------------------------
def rewrite_query(original_query: str) -> str:
    rewrite_prompt = f"""你是一个搜索改写专家。请将用户口语化的提问，改写为1句利于规章/文档检索的标准搜索词。只输出改写后的句子。

【用户原始提问】：{original_query}
【改写后标准问题】："""

    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return original_query


# -------------------------------------------------------------
# 6. 在线模块 C：带置信度阈值过滤 (Score Threshold) 的两阶段检索
# -------------------------------------------------------------
def search_with_confidence_fallback(query: str, recall_k=6, rerank_k=3, score_threshold=0.10):
    """
    两阶段检索 + 低置信度拦截降级逻辑
    """
    # 1. 改写 Query
    search_q = rewrite_query(query)
    print(f"🔄 [Query Rewriter 改写]: '{query}' ──> '{search_q}'")

    # 2. 粗筛召回
    bm25_docs = bm25_retriever.invoke(search_q)
    vec_docs = vector_retriever.invoke(search_q)

    candidate_docs = []
    seen = set()
    for doc in bm25_docs + vec_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            candidate_docs.append(doc)

    # 3. 精排打分
    pairs = [[search_q, doc.page_content] for doc in candidate_docs]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_score = scored_docs[0][1] if scored_docs else -999.0
    print(f"📊 [Reranker 最高精排匹配得分 Top-1 Score]: {top_score:.4f}")

    # 4. 低置信度阈值判定拦截 (Low-Confidence Fallback)
    if top_score < score_threshold:
        print("⚠️ [警告]: 精排匹配得分低于安全阈值，触发低置信度防幻觉降级拒答！")
        return None, top_score

    final_docs = [doc for doc, score in scored_docs[:rerank_k]]
    return final_docs, top_score


# -------------------------------------------------------------
# 7. 全量 RAG 问答交互循环
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== 🚀 RAG V4 (工业级全量多模态检索管线) 已就绪！(输入 exit 退出) ===")

    while True:
        user_input = input("\n请针对知识库提问: ")
        if user_input.strip().lower() == "exit":
            print("问答系统已退出。")
            break

        if not user_input.strip():
            continue

        # 第一步：意图路由分类 (Question Router)
        intent = question_router(user_input)
        print(f"🧭 [Question Router 分类意图]: {intent}")

        if intent == "chitchat":
            # 闲聊直接调用 LLM，不查向量库
            chat_res = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": user_input}]
            )
            print(f"\nAI (闲聊回复): {chat_res.choices[0].message.content}")
            continue

        # 第二步：知识库问答 -> 触发带置信度阈值过滤的两阶段检索
        retrieved_docs, max_score = search_with_confidence_fallback(user_input, score_threshold=0.10)

        if retrieved_docs is None:
            # 置信度过低，直接拒答防幻觉
            print("\nAI (低置信度降级回复): 知识库中未找到相关内容，请尝试提供更明确的提问。")
            continue

        # 第三步：构造生成 Prompt 并显示切片与来源 Metadata
        context_text = "\n\n".join([
            f"[切片{i + 1}] (来源: {doc.metadata.get('filename', '未知')}, 类型: {doc.metadata.get('media_type', 'text')}):\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        ])

        print(f"\n🔍 【检索召回的带来源切片】:\n{context_text}")

        prompt = f"""你是一个严谨的私有知识库助手。请根据下方提供的【参考资料】，回答【用户提问】。

【原则】：
1. 优先提取资料中的具体数字、时间、规则直接回答，并指明信息来源。
2. 如果【参考资料】中完全没有相关信息，回答“知识库中未找到相关内容”。

【参考资料】:
{context_text}

【用户提问】:
{user_input}
"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的私有知识库问答助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        print(f"\nAI (基于检索与引用回复): {response.choices[0].message.content}")