import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import jieba
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# -------------------------------------------------------------
# 1. 基础配置与环境变量加载
# -------------------------------------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    st.error("❌ 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# -------------------------------------------------------------
# 2. 缓存加载重型 Embedding 和 Reranker 模型（避免网页刷新重复加载）
# -------------------------------------------------------------
@st.cache_resource
def load_heavy_models():
    print("⏳ 正在加载本地 Embedding 及 Reranker 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    return embeddings, reranker


embeddings_model, reranker_model = load_heavy_models()


# -------------------------------------------------------------
# 3. 动态构建知识库索引核心函数
# -------------------------------------------------------------
def load_all_documents(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    documents = []

    # 加载 txt 和 md
    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents.extend(txt_loader.load())
    md_loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents.extend(md_loader.load())

    # 加载 pdf
    try:
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"⚠️ PDF 加载提示: {e}")

    return documents


def build_knowledge_base(data_dir="data"):
    docs = load_all_documents(data_dir)
    if not docs:
        return None, None, 0

    # 文本切片
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    # 分词与 BM25
    def chinese_tokenizer(text: str):
        return list(jieba.cut(text))

    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=chinese_tokenizer)
    bm25_retriever.k = 6

    # Chroma 向量库
    vectorstore = Chroma.from_documents(chunks, embeddings_model)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    return bm25_retriever, vector_retriever, len(chunks)


# 初始化 Session State 中的检索器
if "bm25" not in st.session_state or "vector" not in st.session_state:
    with st.spinner("🚀 正在初始化知识库索引，请稍候..."):
        bm25, vec, count = build_knowledge_base("data")
        st.session_state.bm25 = bm25
        st.session_state.vector = vec
        st.session_state.chunk_count = count

# -------------------------------------------------------------
# 4. Streamlit 网页布局与侧边栏文件管理
# -------------------------------------------------------------
st.set_page_config(
    page_title="个人知识库 RAG 问答系统",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 面向个人的 RAG 知识库问答系统")
st.caption("🚀 支持 PDF / Markdown / TXT 网页端拖拽上传与两阶段智能问答")

with st.sidebar:
    st.header("📂 知识库文件上传与管理")

    uploaded_files = st.file_uploader(
        "选择或拖拽文件上传到知识库:",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.toast(f"✅ 成功保存新文件: {uploaded_file.name}", icon="📄")

    if st.button("🔄 实时重新构建索引", type="primary"):
        with st.spinner("正在重新切片并建立混合索引 (BM25 + Chroma) ..."):
            bm25, vec, count = build_knowledge_base("data")
            st.session_state.bm25 = bm25
            st.session_state.vector = vec
            st.session_state.chunk_count = count
            st.success(f"索引重建成功！当前知识库共包含 {count} 个有效切片。")

    st.metric("📦 当前知识库切片数 (Chunks)", st.session_state.get("chunk_count", 0))


# -------------------------------------------------------------
# 5. 核心问答与两阶段检索逻辑
# -------------------------------------------------------------
def search_and_answer(query: str):
    if not st.session_state.bm25 or not st.session_state.vector:
        return None, "知识库为空，请先上传文档并重建索引！"

    # 1. 简单的 Query 改写
    prompt_rw = f"将用户口语化提问改写为1句利于文档检索的标准搜索词，只输出改写后的句子：{query}"
    try:
        res_rw = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt_rw}],
            temperature=0.1
        )
        search_q = res_rw.choices[0].message.content.strip()
    except Exception:
        search_q = query

    # 2. 双路召回
    bm25_docs = st.session_state.bm25.invoke(search_q)
    vec_docs = st.session_state.vector.invoke(search_q)

    candidate_docs = []
    seen = set()
    for doc in bm25_docs + vec_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            candidate_docs.append(doc)

    if not candidate_docs:
        return None, 0.0

    # 3. Reranker 精排
    pairs = [[search_q, doc.page_content] for doc in candidate_docs]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_score = scored_docs[0][1] if scored_docs else 0.0
    final_docs = [doc for doc, score in scored_docs[:3]]

    return final_docs, top_score


# -------------------------------------------------------------
# 6. 聊天对话 UI 界面
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！请在左侧侧边栏上传你的 PDF 或笔记文件，点击刷新索引，然后向我提问！"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("🔍 正在检索并精排知识库..."):
            retrieved_docs, max_score = search_and_answer(user_input)

        if not retrieved_docs:
            ai_reply = "⚠️ **知识库中未找到相关内容**。"
            message_placeholder.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        else:
            with st.expander(f"📌 点击查看精排召回的参考切片 (Top-1 分数: {max_score:.4f})"):
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"**[切片 {i + 1}]** 来源文件: `{doc.metadata.get('source', '未知')}`")
                    st.code(doc.page_content, language="markdown")

            context_text = "\n\n".join([
                f"[切片{i + 1}]:\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            ])

            prompt = f"""你是一个严谨的私有知识库助手。请根据下方提供的【参考资料】，回答【用户提问】。

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

            ai_reply = response.choices[0].message.content
            message_placeholder.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})