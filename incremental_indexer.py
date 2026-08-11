import os
import hashlib
import json
from langchain_community.document_loaders import TextLoader, PyPDFLoader

MANIFEST_FILE = "data_manifest.json"


def calculate_md5(file_path: str) -> str:
    """计算单个文件的 MD5 哈希值"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def get_file_manifest(data_dir="data") -> dict:
    """获取当前 data 目录下所有文件的 MD5 状态"""
    current_manifest = {}
    if not os.path.exists(data_dir):
        return current_manifest

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".txt", ".md", ".pdf", ".png", ".jpg")):
                file_path = os.path.join(root, file)
                current_manifest[file_path] = calculate_md5(file_path)
    return current_manifest


def check_incremental_changes(data_dir="data"):
    """
    增量检测算法：比对 data/ 目录与清单文件 data_manifest.json
    返回: (added_files, modified_files, deleted_files, updated_manifest)
    """
    old_manifest = {}
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            try:
                old_manifest = json.load(f)
            except Exception:
                old_manifest = {}

    new_manifest = get_file_manifest(data_dir)

    added_files = []
    modified_files = []
    deleted_files = []

    # 检测新增与修改
    for path, new_hash in new_manifest.items():
        if path not in old_manifest:
            added_files.append(path)
        elif old_manifest[path] != new_hash:
            modified_files.append(path)

    # 检测删除
    for path in old_manifest:
        if path not in new_manifest:
            deleted_files.append(path)

    return added_files, modified_files, deleted_files, new_manifest


def save_manifest(manifest: dict):
    """保存更新后的文件状态清单"""
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("🔍 正在扫描 data/ 目录的 MD5 文件增量变化 ...")
    added, modified, deleted, new_manifest = check_incremental_changes("data")

    print(f"✨ [新增文件]: {len(added)} 个 -> {added}")
    print(f"🔄 [修改文件]: {len(modified)} 个 -> {modified}")
    print(f"🗑️ [删除文件]: {len(deleted)} 个 -> {deleted}")

    # 更新保存 manifest
    save_manifest(new_manifest)
    print("💾 已更新本地 data_manifest.json 增量清单！")