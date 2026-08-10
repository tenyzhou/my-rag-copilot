import os
import json
from openai import OpenAI
from rag_v3 import two_stage_rag_search  # 复用我们 V3 跑通的两阶段检索函数

# -------------------------------------------------------------
# 1. 初始化 DeepSeek 客户端（兼任 LLM-as-a-Judge 裁判）
# -------------------------------------------------------------
client = OpenAI(
    api_key="sk-3b32024fff8d47f08142528c8f5fdbd3",  # ⚠️ 替换为你申请到的真实 Key
    base_url="https://api.deepseek.com"
)

# -------------------------------------------------------------
# 2. 准备 Benchmark 测试数据集 (包含测试问题与标准参考答案 Ground Truth)
# -------------------------------------------------------------
test_dataset = [
    {
        "question": "团队上班时间和打卡截止时间分别是几点？",
        "ground_truth": "团队上班时间为早上 9:30，打卡截止时间为 9:45。"
    },
    {
        "question": "单次差旅餐补上限是多少钱，发票什么时候提交审核？",
        "ground_truth": "单次差旅餐补上限为 150 元/人，凭发票在每周五提交审核。"
    },
    {
        "question": "提交代码至 GitHub 有什么 Commit 规范吗？",
        "ground_truth": "提交至 GitHub 的 Git Commit Message 必须符合 Conventional Commits 规范，例如 feat: 或 fix: 开头。"
    },
    {
        "question": "公司的紧急联系人是谁，联系电话是多少？",
        "ground_truth": "紧急联系人为技术负责人张三，电话 138-0000-0000。"
    }
]


# -------------------------------------------------------------
# 3. LLM-as-a-Judge 打分核心函数
# -------------------------------------------------------------
def evaluate_rag_triad(question, retrieved_contexts, generated_answer, ground_truth):
    """
    利用 DeepSeek 裁判对 RAG 的三大核心维度进行自动化 0-1.0 量化打分
    """
    eval_prompt = f"""你是一名严格的 RAG 系统评估专家。请根据提供的资料，对系统的表现进行客观打分（得分范围 0.0 到 1.0）。

【评估输入】:
- 用户问题: {question}
- 检索到的上下文 (Contexts): {retrieved_contexts}
- RAG 生成的回答 (Answer): {generated_answer}
- 标准答案 (Ground Truth): {ground_truth}

【请评估以下 3 个指标并严格按 JSON 格式输出】:
1. faithfulness (忠实度): 生成的回答中的事实，有多少能在“检索到的上下文”中找到直接依据？(1.0 为完全有依据，0.0 为完全瞎编)
2. answer_relevance (回答相关性): 生成的回答是否直接、完整地回答了“用户问题”？(1.0 为非常契合，0.0 为完全离题)
3. context_recall (上下文召回率): “检索到的上下文”是否涵盖了“标准答案”里的核心信息点？(1.0 为完全涵盖，0.0 为未召回)

请仅输出一个合法的 JSON 字典，格式如下：
{{
  "faithfulness": 0.95,
  "answer_relevance": 1.0,
  "context_recall": 0.90,
  "reason": "评价简述"
}}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": eval_prompt}],
        response_format={"type": "json_object"}  # 强制结构化 JSON 输出
    )

    return json.loads(response.choices[0].message.content)


# -------------------------------------------------------------
# 4. 执行全自动流水线评估 (Evaluation Pipeline)
# -------------------------------------------------------------
print("=== 🚀 正在启动 RAG V3 系统自动化 RAGAS 量化评估流水线 ===\n")

total_faithfulness = 0
total_answer_relevance = 0
total_context_recall = 0
results_summary = []

for i, item in enumerate(test_dataset):
    q = item["question"]
    gt = item["ground_truth"]

    print(f"正在评估测试用例 [{i + 1}/{len(test_dataset)}]: {q}")

    # 1. 运行 RAG 检索端 (V3 两阶段检索)
    retrieved_docs = two_stage_rag_search(q, top_n_recall=6, top_k_rerank=2)
    contexts_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # 2. 运行 RAG 生成端
    gen_prompt = f"请严格根据资料回答问题。资料：{contexts_text}\n问题：{q}"
    ans_res = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": gen_prompt}]
    )
    generated_ans = ans_res.choices[0].message.content

    # 3. 裁判打分
    scores = evaluate_rag_triad(q, contexts_text, generated_ans, gt)

    total_faithfulness += scores["faithfulness"]
    total_answer_relevance += scores["answer_relevance"]
    total_context_recall += scores["context_recall"]

    results_summary.append({
        "case": i + 1,
        "question": q,
        "faithfulness": scores["faithfulness"],
        "answer_relevance": scores["answer_relevance"],
        "context_recall": scores["context_recall"],
        "reason": scores["reason"]
    })

# -------------------------------------------------------------
# 5. 汇总并打印 RAGAS 评估雷达简报
# -------------------------------------------------------------
num_cases = len(test_dataset)
avg_faithfulness = total_faithfulness / num_cases
avg_answer_relevance = total_answer_relevance / num_cases
avg_context_recall = total_context_recall / num_cases

print("\n" + "=" * 60)
print("📊 【 RAG V3 系统 自动化评估最终成绩单 】")
print("=" * 60)
print(f"🟢 忠 实 度 (Faithfulness)      : {avg_faithfulness:.4f}  (防幻觉能力)")
print(f"🟢 回答相关性 (Answer Relevance)  : {avg_answer_relevance:.4f}  (问答契合度)")
print(f"🟢 上下文召回率 (Context Recall)   : {avg_context_recall:.4f}  (检索覆盖率)")
print("=" * 60)

print("\n📋 各测试用例详细得分明细:")
for res in results_summary:
    print(
        f"用例 {res['case']} | F: {res['faithfulness']} | AR: {res['answer_relevance']} | CR: {res['context_recall']} | 归因: {res['reason']}")