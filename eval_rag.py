import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from rag_v3 import two_stage_rag_search

# 1. 加载本地 .env 环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("❌ 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

# 2. 初始化 DeepSeek 裁判客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# -------------------------------------------------------------
# 3. 工业级高压 Benchmark 测试数据集 (包含边界题、拒答题、噪音题)
# -------------------------------------------------------------
test_dataset = [
    {
        "type": "直接提取题",
        "question": "团队上班时间和打卡截止时间分别是几点？",
        "ground_truth": "团队上班时间为早上 9:30，打卡截止时间为 9:45。"
    },
    {
        "type": "拒答/超纲压力测试",
        "question": "公司午饭包吃吗？有班车接送吗？",
        "ground_truth": "知识库中未找到相关内容。"
    },
    {
        "type": "口语化/错别字噪音测试",
        "question": "老铁，最迟几点得打卡啊？晚了算啥情况？",
        "ground_truth": "打卡截止时间为 9:45。关于迟到扣款等具体情况知识库未提及。"
    },
    {
        "type": "错误前提诱导测试",
        "question": "听说咱们单次餐补上限是 500 元，对吧？",
        "ground_truth": "不对，单次差旅餐补上限为 150 元/人。"
    },
    {
        "type": "跨段落综合提取题",
        "question": "请告诉我紧急联系人张三的电话，以及提交 GitHub 代码的格式要求。",
        "ground_truth": "张三电话为 138-0000-0000；GitHub Commit Message 必须符合 Conventional Commits 规范，如 feat: 或 fix: 开头。"
    }
]


# 4. LLM-as-a-Judge 评估核心函数
def evaluate_rag_triad(question, retrieved_contexts, generated_answer, ground_truth):
    eval_prompt = f"""你是一名严格的 RAG 系统评估专家。请根据提供的资料，对系统的表现进行客观打分（得分范围 0.0 到 1.0）。

【评估输入】:
- 用户问题: {question}
- 检索到的上下文 (Contexts): {retrieved_contexts}
- RAG 生成的回答 (Answer): {generated_answer}
- 标准答案 (Ground Truth): {ground_truth}

【请评估以下 3 个指标并严格按 JSON 格式输出】:
1. faithfulness (忠实度): 生成的回答中的事实，有多少能在“检索到的上下文”中找到直接依据？如果上下文没有写，但模型自己凭空回答了，必须打低分（如 0.0-0.3）！如果模型正确回答“未找到”，打 1.0。
2. answer_relevance (回答相关性): 生成的回答是否直接、完整地回答了“用户问题”？(1.0 为非常契合，0.0 为完全离题)
3. context_recall (上下文召回率): “检索到的上下文”是否涵盖了“标准答案”里的核心信息点？(1.0 为完全涵盖，0.0 为未召回)

请仅输出一个合法的 JSON 字典，格式如下：
{{
  "faithfulness": 0.85,
  "answer_relevance": 0.90,
  "context_recall": 0.80,
  "reason": "评价简述，指出扣分原因"
}}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": eval_prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


# 5. 主程序运行流水线
if __name__ == "__main__":
    print("=== 🚀 正在启动 RAG V3 系统高压 RAGAS 量化评估流水线 ===\n")

    total_faithfulness = 0
    total_answer_relevance = 0
    total_context_recall = 0
    results_summary = []

    for i, item in enumerate(test_dataset):
        q = item["question"]
        gt = item["ground_truth"]
        q_type = item["type"]

        print(f"正在评估用例 [{i + 1}/{len(test_dataset)}] ({q_type}): {q}")

        # 运行两阶段检索
        retrieved_docs = two_stage_rag_search(q, top_n_recall=6, top_k_rerank=2)
        contexts_text = "\n".join([doc.page_content for doc in retrieved_docs])

        # 生成回答
        gen_prompt = f"""你是一个严谨的私有知识库助手。请根据下方提供的【参考资料】，回答【用户提问】。

【原则】：
1. 优先提取资料中的具体数字、时间、规则直接回答。
2. 如果【参考资料】中完全没有相关信息，才回答“知识库中未找到相关内容”。

【参考资料】:
{contexts_text}

【用户提问】:
{q}
"""
        ans_res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": gen_prompt}]
        )
        generated_ans = ans_res.choices[0].message.content

        # 裁判打分
        scores = evaluate_rag_triad(q, contexts_text, generated_ans, gt)

        total_faithfulness += scores["faithfulness"]
        total_answer_relevance += scores["answer_relevance"]
        total_context_recall += scores["context_recall"]

        results_summary.append({
            "case": i + 1,
            "type": q_type,
            "question": q,
            "faithfulness": scores["faithfulness"],
            "answer_relevance": scores["answer_relevance"],
            "context_recall": scores["context_recall"],
            "reason": scores["reason"]
        })

    num_cases = len(test_dataset)
    avg_faithfulness = total_faithfulness / num_cases
    avg_answer_relevance = total_answer_relevance / num_cases
    avg_context_recall = total_context_recall / num_cases

    print("\n" + "=" * 60)
    print("📊 【 RAG V3 系统 高压评估最终成绩单 】")
    print("=" * 60)
    print(f"🟢 忠 实 度 (Faithfulness)      : {avg_faithfulness:.4f}  (防幻觉能力)")
    print(f"🟢 回答相关性 (Answer Relevance)  : {avg_answer_relevance:.4f}  (问答契合度)")
    print(f"🟢 上下文召回率 (Context Recall)   : {avg_context_recall:.4f}  (检索覆盖率)")
    print("=" * 60)

    print("\n📋 各测试用例详细得分与扣分归因明细:")
    for res in results_summary:
        print(
            f"用例 {res['case']} [{res['type']}] | F: {res['faithfulness']} | AR: {res['answer_relevance']} | CR: {res['context_recall']}")
        print(f"   💡 扣分归因: {res['reason']}\n")