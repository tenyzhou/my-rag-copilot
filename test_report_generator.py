import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 导入阶段 2 的检索能力
from test_recommender import bm25_retriever, vector_retriever, reranker_model, client

# -------------------------------------------------------------
# 1. 测试报告自动生成核心引擎
# -------------------------------------------------------------
def generate_hardware_test_report(product_name: str, raw_measurements: dict) -> str:
    """
    【xTool 核心职责 2 实现】: 输入硬件测量真实数据，自动比对 RAG 阈值，生成含 Pass/Fail 判定与根因分析的报告
    """
    print(f"\n📊 [报告生成引擎启动]: 正在处理『{product_name}』的实测数据，匹配判定规则...")

    # 1. 召回相关的硬件测试规则
    query = f"{product_name} 功率 温升 判定合格标准"
    bm25_docs = bm25_retriever.invoke(query)
    vec_docs = vector_retriever.invoke(query)

    candidate_docs = []
    seen = set()
    for doc in bm25_docs + vec_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            candidate_docs.append(doc)

    pairs = [[query, doc.page_content] for doc in candidate_docs]
    scores = reranker_model.predict(pairs)
    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    rules_context = "\n\n".join([doc.page_content for doc, score in scored_docs[:2]])

    # 2. 构造 Prompt 指导 LLM 比对数据与进行归因诊断
    system_prompt = """你现在是创客工场(xTool)硬件质量与测试工程专家。
你的任务是根据传入的【硬件测试标准规范】和工程师上传的【实测数据】，自动比对指标并出具正式的《xTool 硬件研发自测终审报告》。

【生成要求】：
1. **自动数据比对**：将【实测数据】逐项与【合格标准】做比对，明确标注数值，并打上 【PASS】 或 【FAIL】 标记。
2. **智能异常归因（关键加分项）**：如果有任意一项测试指标为 FAIL（不合格），你必须结合硬件/机械工程经验，提供至少 2 条合理的“超标原因排查方向与改进建议”。
3. **输出格式**：格式必须工整，包含：报告基本信息、测试结果汇总表、详细数据对比表、异常归因分析（若有FAIL）、终审结论与签字栏。
"""

    measurements_str = json.dumps(raw_measurements, ensure_ascii=False, indent=2)

    user_prompt = f"""【被测产品名称】: {product_name}

【硬件测试标准规范 (合格阈值来源)】:
{rules_context}

【工程师实验室实测原始数据】:
{measurements_str}

请自动比对数据并生成专业的《xTool 硬件研发自测终审报告》："""

    print("🤖 正在调用 LLM 进行指标自动化比对、Pass/Fail 判定与异常归因诊断...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


# -------------------------------------------------------------
# 2. 模拟实验室真实测试场景验证
# -------------------------------------------------------------
if __name__ == "__main__":
    # 模拟一份含有“芯片温度超标”的真实测试数据
    mock_lab_data = {
        "测试时间": "2026-08-11",
        "测试人员": "天一 (硬件测试工程师)",
        "环境温度": "24.0℃",
        "环境湿度": "58% RH",
        "TS-LASER-001 功率与光斑测试": {
            "起始输出功率_P0": "39.8 W",
            "30分钟持续输出功率_P30": "38.9 W",
            "功率波动率": "2.1%",
            "聚焦光斑尺寸": "0.07mm x 0.09mm"
        },
        "TS-LASER-002 散热与温升测试": {
            "散热片最高温度_T2": "61.2 ℃",
            "驱动芯片表面最高温度_T1": "89.5 ℃",  # ⚠️ 故意传入超标数据（标准为 ≤ 85℃），测试 AI 的归因诊断能力！
            "120分钟内过热保护断电": "否"
        }
    }

    report = generate_hardware_test_report("40W 激光雕刻头模组", mock_lab_data)

    print("\n================ 📄 xTool 自动化测试报告与归因分析预览 ================")
    print(report)