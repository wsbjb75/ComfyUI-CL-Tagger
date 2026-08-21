[README.md](https://github.com/user-attachments/files/31306742/README.md)
# ComfyUI-CL-Tagger-v2

适配 [cella110n/cl_tagger_v2](https://huggingface.co/cella110n/cl_tagger_v2) 的独立 ComfyUI 插件。给图像自动生成 Danbooru 风格标签（10.8 万标签），常用于反推提示词、批量打标、训练集自动加标签。

## 特性

- 完整复刻官方后处理：**logits → sigmoid → 100 桶直方图校准查表**（`model_tag_metrics.npz` 的 `calibration_table`，Jeffreys 平滑 `(pos+0.5)/(total+1)` 生成），输出的是校准后概率而非原始 logits
- 支持各标签训练集最优 F1 阈值（`best_thr`）一键切换
- 基于 onnxruntime，CPU / CUDA / DirectML 自动选择 provider
- 自动扫描模型目录，支持排除词、下划线替换、按置信度/字母排序、附带分数

## 安装

1. 将本目录复制到 `ComfyUI/custom_nodes/ComfyUI-CL-Tagger-v2`
2. 安装依赖：`pip install onnxruntime`（torch / numpy / Pillow 由 ComfyUI 自带）
3. 模型文件放入 `<ComfyUI>/models/onnx/cl_tagger/`（插件自动注册该目录）：

```
models/onnx/cl_tagger/
├── model.onnx              推理模型（约 0.76 MB）
├── model.onnx.data         外部权重（约 2.1 GB，必须与 model.onnx 同目录）
├── model_vocabulary.json   标签词表 tag_to_idx（108139 标签）
└── model_tag_metrics.npz   校准数据（calibration_table / best_thr）
```

也可以运行 `python download.py` 自动下载到上述目录。
4. 重启 ComfyUI

## 使用

节点位置：`image/tagger` → **CL Tagger v2 (标签反推)**

工作流示例：

```
LoadImage ──► CLTaggerV2 ──► tags (STRING) ──► CLIPTextEncode / SaveText
```

### 输入

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| image | 输入图像（IMAGE） | - |
| model | 模型文件（自动扫描，多模型下拉切换） | - |
| threshold | 标签置信度阈值，作用于**校准后概率**。0.5 附近较均衡，调低输出更多标签 | 0.50 |
| character_threshold | **角色类标签**的置信度阈值。cl_tagger_v2 词表没有 WD14 那种角色分类列，角色判定规则见下 | 0.65 |
| exclude_tags | 排除标签，逗号分隔（如 `watermark, signature`） | 空 |
| tag_replacement | 标签内下划线的替换字符，填空格可转为自然语言提示词 | `_` |
| sort_by | 排序：confidence / alphabetical / none | confidence |
| with_scores | 标签串中附带校准概率 | false |
| character_tags | 自定义角色名单，逗号分隔（如 `hatsune miku, kagamine rin`），名单内标签按 character_threshold 过滤 | 空 |
| use_best_threshold | 启用各标签训练集最优 F1 阈值（best_thr，分布约 0.4~0.65） | false |

### 角色标签判定（character_threshold 的适用范围）

由于 cl_tagger_v2 的 `model_vocabulary.json` 只有标签名、没有角色分类列，插件用以下两条规则判定角色类标签，命中其一即按 `character_threshold` 过滤：

1. **自定义名单**：`character_tags` 参数里列出的标签（忽略大小写与下划线/空格差异），如 `hatsune miku` 这类不带括号的角色名，建议在此补充
2. **括号消歧义**：danbooru 角色名惯例「名字 (作品名)」形式，如 `kagamine rin (len)`、`asuka langley soryu (evangelion)`、`shouta aizawa (mha)`

### 输出

| 输出 | 类型 | 内容 |
| --- | --- | --- |
| tags | STRING | 逗号分隔的标签串 |
| tags_with_scores | STRING | 带校准概率的标签串 |
| count | INT | 命中的标签数量 |
| scores | FLOAT[] | 各标签校准概率（与 tags 顺序一致） |

## 技术说明

- **架构**：SigLIP2（`google/siglip2-so400m-patch14-384`）视觉塔 + 108139 标签线性头
- **预处理**：resize 384×384（LANCZOS），归一化 `(x/255 - 0.5) / 0.5`
- **后处理**：sigmoid 概率 `s` → `bin = clip(int(s*100), 0, 99)` → `cal = calibration_table[tag, bin]`
- 校准表加载时已把 99 号低样本噪声桶平滑为 98 号桶的值
- `model_ood_ref.npz`（1152 维嵌入的 OOD 马氏距离参考）为训练侧文件：本模型 ONNX 只输出 logits，不输出嵌入，故运行时 OOD 检测未启用，文件不影响使用

## 常见问题

**节点列表显示"(未找到模型)"？**
确认 `model.onnx` 及配套文件都在 `models/onnx/cl_tagger/` 下（旧版布局 `models/cl_tagger_v2/` 也兼容），并重启了 ComfyUI。`.onnx.data` 必须与 `.onnx` 同目录，否则加载失败。

**输出标签明显不对？**
先确认图是目标域内容（模型在 danbooru 系数据上训练）；阈值低于 0.5 会混入噪声标签，可配合 `use_best_threshold` 对比。


## 目录结构

```
ComfyUI-CL-Tagger-v2/
├── __init__.py       # 节点注册
├── nodes.py          # 主节点（预处理 / 推理 / 校准查表 / 过滤 / 输出）
├── model_utils.py    # 模型目录、词表、校准数据加载与会话管理
├── download.py       # 模型一键下载脚本
├── requirements.txt
└── README.md
```
