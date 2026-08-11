import os
import re
from langchain_core.documents import Document

# -------------------------------------------------------------
# 1. 硬件测试规范 Mock 数据源 (自动容错生成)
# -------------------------------------------------------------
XTOOL_TEST_SPEC_CONTENT = """# xTool 激光雕刻机与智能模组研发测试标准规范（2026版）

## 一、 激光头模组（Laser Module）测试规范
### 1. 激光输出功率与光斑质量测试
- **测试项 ID**: TS-LASER-001
- **适用模组**: 20W/40W/55W 二极管激光头、CO2 激光模组
- **测试环境要求**: 环境温度 23±2℃，湿度 <65% RH
- **测试步骤**: 
  1. 将激光头固定在光功率计正上方 50mm 处；
  2. 预热 3 分钟后，开启 100% 满功率输出，持续照射 60 秒；
  3. 使用光斑分析仪测量焦平面 X/Y 轴光斑直径。
- **判定合格标准**:
  - 40W 模组实际功率 ≥ 38.5W，功率波动率 ≤ ±3%；
  - 聚焦光斑尺寸 ≤ 0.08mm × 0.10mm；
  - 连续工作 30 分钟无热衰减（功率下降 ≤ 5%）。

### 2. 激光头散热与温升测试
- **测试项 ID**: TS-LASER-002
- **适用模组**: 所有功率等级的激光雕刻头
- **测试步骤**: 贴附热电偶传感器于风扇散热片及驱动芯片表面，满功率连续雕刻 120 分钟，记录温度曲线。
- **判定合格标准**: 驱动芯片表面最高温度 ≤ 85℃，散热片温度 ≤ 65℃，无过热保护断电现象。

---

## 二、 X/Y 轴传动与步进电机模组测试规范
### 1. 高速雕刻重复定位精度测试
- **测试项 ID**: TS-MOT-001
- **适用产品**: xTool D1/S1/P2 激光雕刻机全系列机型
- **测试步骤**: 使用千分表固定在移动滑块端，驱动 X/Y 轴以 400mm/s 速度往复运动 500 次。
- **判定合格标准**: X 轴重复定位误差 ≤ ±0.02mm，Y 轴重复定位误差 ≤ ±0.03mm，同步带无明显松弛与打滑。

---

## 三、 智能抽风与整机安全防护测试规范
### 1. 烟雾警报与气流风压测试
- **测试项 ID**: TS-SAF-001
- **适用产品**: 密闭式激光切割机及配套烟雾净化器
- **测试步骤**: 在加工舱内引入标准测试烟雾，检测传感器响应时间及负压差。
- **判定合格标准**: 烟雾传感器在 1.5 秒内触发报警并自动切断激光；加工舱内部负压 ≥ 120 Pa。
"""


# -------------------------------------------------------------
# 2. 结构化 Markdown 测试用例解析函数
# -------------------------------------------------------------
def parse_test_cases_from_markdown(md_content: str, source_filename: str) -> list[Document]:
    """
    【工程化解析器】: 将非结构化的 Markdown 测试规范，提取为带有结构化 Metadata 的 Document 对象
    """
    documents = []
    # 使用正则表达式按 ### 标记拆分独立测试用例
    sections = re.split(r'\n###\s+', md_content)

    main_module = "通用硬件模组"

    for section in sections:
        if section.startswith('#'):
            match = re.search(r'#+\s+(.*)', section)
            if match:
                main_module = match.group(1).strip()
            continue

        lines = section.strip().split('\n')
        test_item_title = lines[0].strip() if lines else "未知测试项"

        # 匹配 TS-LASER-001 这类测试用例编号
        test_id_match = re.search(r'TS-[A-Z]+-\d+', section)
        test_id = test_id_match.group(0) if test_id_match else "TS-GENERIC"

        # 匹配适用模组
        module_match = re.search(r'适用模组[：:]\s*(.*)', section)
        applied_module = module_match.group(1).strip() if module_match else main_module

        metadata = {
            "source": source_filename,
            "test_id": test_id,
            "module_category": main_module,
            "applied_module": applied_module,
            "title": test_item_title,
            "media_type": "test_spec"
        }

        doc = Document(
            page_content=f"测试项名称: {test_item_title}\n测试用例编号: {test_id}\n适用模组: {applied_module}\n\n{section}",
            metadata=metadata
        )
        documents.append(doc)

    print(f"📊 [工程化解析完成]: 成功提取出 {len(documents)} 个结构化测试用例节点！")
    return documents


def ensure_data_file():
    """数据管线自愈函数：若文件不存在自动创建并写入数据"""
    data_dir = "data"
    file_path = os.path.join(data_dir, "xtool_laser_test_spec.md")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(XTOOL_TEST_SPEC_CONTENT)
        print(f"✅ [数据管线自愈]: 已自动创建 {file_path} 并写入硬件测试规范数据！")

    return file_path


# -------------------------------------------------------------
# 3. 运行测试
# -------------------------------------------------------------
if __name__ == "__main__":
    spec_path = ensure_data_file()
    with open(spec_path, "r", encoding="utf-8") as f:
        content = f.read()
    docs = parse_test_cases_from_markdown(content, "xtool_laser_test_spec.md")

    if docs:
        print("\n✅ 解析出的第一个结构化测试用例 Preview:")
        print(f"   🆔 用例编号: {docs[0].metadata['test_id']}")
        print(f"   🏷️ 适用模组: {docs[0].metadata['applied_module']}")
        print(f"   📄 规则预览:\n{docs[0].page_content[:150]}...")