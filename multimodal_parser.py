import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key if api_key else "dummy_key",
    base_url="https://api.deepseek.com"
)

# 2. 初始化本地 RapidOCR 识别引擎
ocr_engine = RapidOCR()


def process_image_file(image_path: str) -> Document:
    """
    多模态图片处理核心管线：
    1. 使用 RapidOCR 提取图片中的所有文字
    2. 使用 LLM/VLM 将文字和图表结构进行语义润色与描述生成
    3. 封装为带丰富的 Metadata 元数据的 LangChain Document
    """
    print(f"🖼️ 正在解析图片文件: {image_path} ...")

    # 步骤 A: 运行 OCR 识别文字
    ocr_result, _ = ocr_engine(image_path)
    ocr_text_lines = []
    if ocr_result:
        for item in ocr_result:
            ocr_text_lines.append(item[1])  # 提取识别出的文本内容

    raw_ocr_text = "\n".join(ocr_text_lines) if ocr_text_lines else "（图片中未识别出清晰文字）"
    print(f"   └─ 🔍 [RapidOCR 提取文字]: {raw_ocr_text[:60]}...")

    # 步骤 B: 使用 LLM 对 OCR 提取的碎片文字进行 VLM 语义描述化（图片字幕生成）
    vlm_caption_prompt = f"""你是一个多模态图像与文档解析助手。请根据下方从一张知识库图片/架构图中用 OCR 提取出的碎片文字，总结该图片表达的核心含义与架构逻辑。

【OCR 原始提取文字】：
{raw_ocr_text}

【要求】：
1. 给出一段 100 字左右的结构化图像语义描述（如：这是一张关于XX系统的架构图，包含XX核心组件等）。
2. 保留原图中的关键专有名词、流程节点和电话/数字信息。
3. 直接输出描述文本，不要带有任何多余的引言。
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": vlm_caption_prompt}],
            temperature=0.2
        )
        image_caption = response.choices[0].message.content.strip()
    except Exception as e:
        image_caption = f"知识库图像节点，包含识别文字: {raw_ocr_text}"

    # 步骤 C: 组合成包含全面 Metadata 元数据的完整 Document
    full_content = f"【图像语义描述】:\n{image_caption}\n\n【图片包含文字(OCR)】:\n{raw_ocr_text}"

    metadata = {
        "source": image_path,
        "filename": os.path.basename(image_path),
        "media_type": "image",  # 元数据绑定：媒体类型为图片
        "has_ocr": True,  # 元数据绑定：已完成 OCR 识别
        "ocr_char_count": len(raw_ocr_text)
    }

    doc = Document(page_content=full_content, metadata=metadata)
    return doc


if __name__ == "__main__":
    # 测试扫描 data/ 目录下的一张图片（如果没有图片，可以放一张包含文字或流程图的 png/jpg 进去测试）
    data_dir = "data"
    test_images = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                test_images.append(os.path.join(root, file))

    if not test_images:
        print("💡 提示: data/ 目录下暂无图片文件。你可以放入一张带文字的图片（如 .png/.jpg）再运行测试！")
    else:
        for img_path in test_images:
            parsed_doc = process_image_file(img_path)
            print("\n✅ [解析生成的带 Metadata 多模态 Document]:")
            print(f"   📄 内容 Preview: {parsed_doc.page_content[:150]}...")
            print(f"   🏷️ 元数据 (Metadata): {parsed_doc.metadata}\n")