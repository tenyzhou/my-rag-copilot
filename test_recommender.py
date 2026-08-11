import os
import jieba
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# 导入阶段 1 写好的解析器与自愈函数
from test_case_parser import parse_test_cases_from_markdown, ensure_data_file

# -------------------------------------------------------------
# 1. 环境初始化
# -------------------------------------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("❌ 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    max_retries=3,
    timeout=30.0
)

# -------------------------------------------------------------
# 2. 构建 xTool 硬件测试专用的双路检索器
# -------------------------------------------------------------
print("⏳ 正在加载硬件测试规范并构建本地向量与 BM25 索引...")
spec_file_path = ensure_data_file()

with open(spec_file_path, "r", encoding="utf-8") as f:
    spec_content = f.read()

# 结构化提取 Documents (带用例编号、适用模组等 Metadata)
test_docs = parse_test_cases_from_markdown(spec_content, "xtool_laser_test_spec.md")


# BM25 分词
def chinese_tokenizer(text: str):
    return list(jieba.cut(text))


bm25_retriever = BM25Retriever.from_documents(test_docs, preprocess_func=chinese_tokenizer)
bm25_retriever.k = 5

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(test_docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

reranker_model = CrossEncoder("BAAI/bge-reranker-base")


# -------------------------------------------------------------
# 3. 智能推荐引擎核心逻辑
# -------------------------------------------------------------
def recommend_test_plan(hardware_module_input: str) -> str:
    """
    输入硬件产品/模组名称（如："40W激光头模组"、"X/Y轴传动组"）
    输出：符合 xTool 研发规范的标准化《硬件自测推荐方案》
    """
    print(f"\n🔍 [推荐引擎发起检索]: 正在为目标模组『{hardware_module_input}』匹配自测用例...")

    # 1. 查询改写：将用户输入的模组名优化为测试规范词
    rewrite_prompt = f"""你是一个硬件测试专家。请将用户输入的硬件产品/模组名称，改写为1句利于检索测试用例规范的标准词。只输出改写后的句子。
【用户输入模组】：{hardware_module_input}
【改写测试搜索词】："""

    try:
        rw_res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.1
        )
        search_q = rw_res.choices[0].message.content.strip()
    except Exception:
        search_q = hardware_module_input

    # 2. 混合召回 + Reranker 精排
    bm25_candidates = bm25_retriever.invoke(search_q)
    vec_candidates = vector_retriever.invoke(search_q)

    candidate_docs = []
    seen = set()
    for doc in bm25_candidates + vec_candidates:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            candidate_docs.append(doc)

    pairs = [[search_q, doc.page_content] for doc in candidate_docs]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 选取精排 Top-3 匹配测试项
    top_test_docs = [doc for doc, score in scored_docs[:3]]

    # 3. 构造组装规则上下文
    retrieved_spec_context = "\n------------------\n".join([
        f"【测试项ID】: {doc.metadata.get('test_id')}\n"
        f"【适用范围】: {doc.metadata.get('applied_module')}\n"
        f"【详细内容】:\n{doc.page_content}"
        for doc in top_test_docs
    ])

    # 4. LLM 结构化方案推荐 Prompt
    system_prompt = """你现在是创客工场(xTool)硬件研发测试架构师。
你的任务是根据传入的【测试规范文档切片】，为测试工程师自动匹配并推荐必测的《硬件自测方案》。

【生成要求】：
1. 输出格式必须清晰规范，包含：测试方案名称、推荐测试项列表（含用例编号ID）、测试环境要求、测试步骤与判定合格标准（含硬性数字指标）。
2. 严格基于传入的规范，不得编造不存在的用例编号和数字阈值。
3. 标注每一项测试的硬性合格指标（Pass/Fail 阈值）。
"""

    user_prompt = f"""【研发待测目标产品/模组】: {hardware_module_input}

【检索匹配到的参考自测规范】:
{retrieved_spec_context}

请生成结构化的《xTool 硬件自测推荐方案表》："""

    print("🤖 正在调用 LLM 生成结构化自测推荐方案...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------------------------------------------
# 4. 运行验证
# -------------------------------------------------------------
if __name__ == "__main__":
    # 测试场景 1：测试 40W 激光头模组
    test_input = "准备自测 40W 激光切割头，需要测功率和温升"
    recommendation = recommend_test_plan(test_input)

    print("\n================ 🚀 xTool 智能测试推荐结果 ================")
    print(recommendation)