import os
import json
import streamlit as st
from dotenv import load_dotenv

# 导入我们前三阶段写好的核心模块
from test_case_parser import ensure_data_file
from test_recommender import recommend_test_plan
from test_report_generator import generate_hardware_test_report

# -------------------------------------------------------------
# 1. 页面基础配置
# -------------------------------------------------------------
st.set_page_config(
    page_title="xTool 智能硬件研发测试 AI 工程化平台",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ xTool 智能硬件研发测试 AI 工程化平台")
st.caption("🚀 创客工场 (xTool) 研发测试专属 AI 工具链：基于 RAG + LLM 的测试项智能推荐与报告自动化填充系统")

# 确保测试规范数据集就绪
spec_path = ensure_data_file()

# -------------------------------------------------------------
# 2. 侧边栏：平台状态与参数控制
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 平台状态控制台")
    st.success("✅ 硬件测试规范库：已挂载 `xtool_laser_test_spec.md`")
    st.info("💡 提示：本平台专为 xTool 激光雕刻机/智能模组测试工程师设计，实现测试项秒级推荐与报告智能生成。")

    st.divider()
    st.markdown("### 📦 当前知识库版本")
    st.code("xTool 研发测试标准 2026.V1", language="text")

# -------------------------------------------------------------
# 3. 主界面：三 Tab 多功能解耦工作台
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📚 研发测试规范知识库",
    "🎯 测试项智能推荐器",
    "📄 测试报告自动生成与异常诊断"
])

# -------------------------------------------------------------
# Tab 1: 研发测试规范知识库展示
# -------------------------------------------------------------
with tab1:
    st.subheader("📚 硬件测试规范与判定阈值标准")
    st.markdown("系统当前已索引的标准化自测用例与合格阈值（可实时预览底层结构化 Markdown）：")

    if os.path.exists(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_content = f.read()
        st.text_area("硬件测试规范文本 (Markdown 格式)", spec_content, height=400)
    else:
        st.warning("⚠️ 未找到硬件测试规范数据文件！")

# -------------------------------------------------------------
# Tab 2: 测试项智能推荐器 (兑现 JD 核心职责 1)
# -------------------------------------------------------------
with tab2:
    st.subheader("🎯 硬件自测项智能推荐引擎")
    st.markdown("输入研发待测的产品或模组信息，AI 自动匹配测试用例标准并生成《硬件自测推荐方案表》：")

    col1, col2 = st.columns([2, 1])
    with col1:
        hardware_input = st.text_input(
            "输入待测硬件模组或测试需求：",
            value="准备自测 40W 激光切割头，需要测功率和温升"
        )
    with col2:
        st.write("")  # 占位
        st.write("")
        btn_recommend = st.button("🚀 智能推荐测试方案", type="primary")

    if btn_recommend:
        with st.spinner("🔍 正在检索硬件规范 ➔ 匹配用例 ID ➔ 组合推荐方案..."):
            recommendation_result = recommend_test_plan(hardware_input)

            st.success("✅ 自测方案推荐生成成功！")
            st.markdown(recommendation_result)

            # 提供 Markdown 方案下载按钮
            st.download_button(
                label="📥 导出 Markdown 自测推荐方案",
                data=recommendation_result,
                file_name="xTool_Hardware_Test_Plan.md",
                mime="text/markdown"
            )

# -------------------------------------------------------------
# Tab 3: 测试报告自动生成与异常归因 (兑现 JD 核心职责 2)
# -------------------------------------------------------------
with tab3:
    st.subheader("📄 测试报告数据填充与智能归因诊断")
    st.markdown("输入实验室原始实测数据，系统自动对比合格阈值、判定 Pass/Fail，并对超标项输出根因诊断：")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("##### 1. 输入实验室原始测试数据 (JSON 格式)")
        product_name_input = st.text_input("被测产品型号：", value="40W 激光雕刻头模组")

        # 默认 Mock 一份含超标的数据
        default_lab_json = {
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
                "驱动芯片表面最高温度_T1": "89.5 ℃",
                "120分钟内过热保护断电": "否"
            }
        }

        json_input_str = st.text_area(
            "修改实测数值：",
            value=json.dumps(default_lab_json, ensure_ascii=False, indent=2),
            height=300
        )

        btn_generate_report = st.button("📊 生成终审测试报告", type="primary")

    with col_b:
        st.markdown("##### 2. AI 自动比对判定与报告预览")
        if btn_generate_report:
            try:
                parsed_json = json.loads(json_input_str)
                with st.spinner("🤖 正在比对规范阈值 ➔ 判定 Pass/Fail ➔ 智能推理异常归因..."):
                    generated_report = generate_hardware_test_report(product_name_input, parsed_json)

                    st.success("✅ 测试报告生成完毕！")
                    st.markdown(generated_report)

                    # 下载导出的完整报告
                    st.download_button(
                        label="📥 一键导出 Markdown 正式测试报告",
                        data=generated_report,
                        file_name="xTool_Hardware_Test_Report.md",
                        mime="text/markdown"
                    )
            except json.JSONDecodeError:
                st.error("❌ 输入的 JSON 数据格式有误，请检查语法！")
        else:
            st.info("👈 请在左侧确认实测数据，并点击『生成终审测试报告』按钮。")