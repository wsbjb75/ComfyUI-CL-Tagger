"""一键下载 cl_tagger_v2 模型与词表到 ComfyUI/models/onnx/cl_tagger/。

用法：
    python download.py [--repo cella110n/cl_tagger_v2] [--comfyui-dir <ComfyUI 根目录>]

说明：
    - 需要联网，且已安装 huggingface_hub（pip install huggingface_hub）。
    - 默认自动探测 ComfyUI 根目录（当前目录或其上级目录含 ComfyUI.exe / main.py 时）。
    - 若未探测到，请用 --comfyui-dir 显式指定，或手动把文件放入
      ComfyUI/models/onnx/cl_tagger/ 目录。
"""

import argparse
import os
import sys


def find_comfyui_root(start=None):
    """向上探测包含 main.py + comfy 子目录的 ComfyUI 根目录。"""
    start = start or os.getcwd()
    cur = os.path.abspath(start)
    while True:
        if (os.path.isfile(os.path.join(cur, "main.py"))
                and os.path.isdir(os.path.join(cur, "comfy"))):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def main():
    parser = argparse.ArgumentParser(description="下载 cl_tagger_v2 模型到 ComfyUI")
    parser.add_argument("--repo", default="cella110n/cl_tagger_v2",
                        help="HuggingFace 仓库名（默认 cella110n/cl_tagger_v2）")
    parser.add_argument("--comfyui-dir", default=None,
                        help="ComfyUI 根目录（可选，默认自动探测）")
    parser.add_argument("--local-dir", default=None,
                        help="直接指定模型存放目录（优先于 --comfyui-dir）")
    args = parser.parse_args()

    local_dir = args.local_dir
    if not local_dir:
        root = args.comfyui_dir or find_comfyui_root()
        if not root:
            print("[错误] 未自动探测到 ComfyUI 根目录，请用 --comfyui-dir 指定。",
                  file=sys.stderr)
            sys.exit(1)
        local_dir = os.path.join(root, "models", "onnx", "cl_tagger")

    os.makedirs(local_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[错误] 需要 huggingface_hub：pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"下载 {args.repo} -> {local_dir}")
    snapshot_download(repo_id=args.repo, local_dir=local_dir)
    print("完成。重启 ComfyUI 后即可在节点列表中找到 CLTaggerV2。")


if __name__ == "__main__":
    main()
