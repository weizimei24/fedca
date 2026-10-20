> 2026-10-20：Commonsense、zsRE 和 MetaMathQA 已新增统一训练与评估适配，见 [统一训练说明](README.shared-training.md)。下文记录原数据准备阶段的处理与来源，阶段性“尚未接入”描述请以新说明为准。

# 两组推理实验的数据准备

完成范围：下载固定版本原始数据、来源核对、清洗、客户端划分、seed 42 抽样、评测数据准备及校验。未启动训练或评估，未修改训练流程、模型、优化器、学习率、轮数、LoRA 参数或训练配置。

## 目录约定

- `data/`：保留下载的原始字节、上游说明/许可、固定版本和哈希清单。
- `dataset/`：项目字段规范化后的完整清洗版与每客户端至多 4000 条抽样版。
- `tools/`：独立的数据下载、处理、校验脚本。

| 实验 | 版本 | 客户端数 | 训练样本数 | 目录 |
|---|---|---:|---:|---|
| Commonsense170K | 完整清洗版 | 8 | 147,607 | [commonsense170k_full_seed42](dataset/commonsense170k_full_seed42/README.md) |
| Commonsense170K | 每客户端至多 4000 条 | 8 | 27,365 | [commonsense170k_client4000_seed42](dataset/commonsense170k_client4000_seed42/README.md) |
| MetaMathQA | 完整清洗版 | 8 | 385,676 | [metamathqa_full_seed42](dataset/metamathqa_full_seed42/README.md) |
| MetaMathQA | 每客户端至多 4000 条 | 8 | 32,000 | [metamathqa_client4000_seed42](dataset/metamathqa_client4000_seed42/README.md) |

“完整清洗版”指对完整上游训练集检查后保留的全部可用样本，不是直接复制原始条数。原始 Commonsense170K 170,420 条与 MetaMathQA 395,000 条完整保留在 `data/`。所有剔除都有原始行号、稳定 sample_id 与原因。

## 常识实验

来源：[LLM-Adapters 作者发布的 Commonsense170K](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/816657208af4db747803f87ba40a4c71383fed7a)。客户端固定为：

| 客户端 | 来源 | 原始训练数 | 完整清洗版 | 抽样版 |
|---:|---|---:|---:|---:|
| 0 | boolq | 9,427 | 9,427 | 4,000 |
| 1 | piqa | 16,113 | 16,105 | 4,000 |
| 2 | social_i_qa | 33,410 | 33,369 | 4,000 |
| 3 | hellaswag | 39,905 | 39,905 | 4,000 |
| 4 | winogrande | 63,238 | 40,484 | 4,000 |
| 5 | arc_easy | 2,251 | 2,249 | 2,249 |
| 6 | arc_challenge | 1,119 | 1,116 | 1,116 |
| 7 | openbookqa | 4,957 | 4,952 | 4,000 |

ARC 来源不足 4000 条时保留全部，不放回抽样补数。其余客户端按固定 seed 抽样，抽样版是完整清洗版的严格子集。

常识清洗剔除 22,813 条：22,783 条重复、16 条答案冲突、10 条正确选项歧义、4 条与完整指令评测题重叠。原始 WinoGrande 中的重复记录占主要部分。
作者聚合文件与八个来源训练文件已按样本多重集核对一致。布尔、两选、三选、四选与少量五选题均保留原标签，未强制改成统一 A/B/C/D。

评测数据有两种明确区分的表示：

- `test/0.jsonl` 等：作者发布的评测指令，便于后续项目内的数据加载。
- `benchmarks/<task>.jsonl`：包含 BoolQ passage 等原始 benchmark 字段，供后续 harness 接入。

评测样本全部保留，共 22,419 条；BoolQ、PIQA、SocialIQA、HellaSwag、WinoGrande 使用有标签的 validation，ARC-Easy、ARC-Challenge、OpenBookQA 使用 test。没有把无标签的公开测试集当作可评分数据。两种提示表示的结果不能混称同一评测协议。

## 数学实验

来源：[meta-math/MetaMathQA 官方数据](https://huggingface.co/datasets/meta-math/MetaMathQA/tree/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18)。按原始 `type` 划分为 8 个客户端：

| 客户端 | 来源及增强类型 | 原始训练数 | 完整清洗版 | 抽样版 |
|---:|---|---:|---:|---:|
| 0 | GSM_AnsAug | 80,000 | 75,654 | 4,000 |
| 1 | GSM_Rephrased | 80,000 | 78,917 | 4,000 |
| 2 | GSM_FOBAR | 40,000 | 39,483 | 4,000 |
| 3 | GSM_SV | 40,000 | 39,655 | 4,000 |
| 4 | MATH_AnsAug | 75,000 | 72,904 | 4,000 |
| 5 | MATH_Rephrased | 50,000 | 49,549 | 4,000 |
| 6 | MATH_FOBAR | 15,000 | 14,780 | 4,000 |
| 7 | MATH_SV | 15,000 | 14,734 | 4,000 |

数学清洗剔除 9,324 条：9,320 条同一增强类型中的重复提示/解答、4 条空文本；本次精确匹配检查未发现 query 或 original_question 与官方 GSM8K/MATH 测试题重叠。
保留完整推理和 original_question；不因多种增强共享一道原始训练题而删掉不同解答。数学比较只折叠空白，区分 X/x、\Pi/\pi；原始训练文本不做改写。所有有效原始问题均匹配到下载的 GSM8K/MATH 官方训练问题。

评测集：

- `evaluation/gsm8k.jsonl`：[OpenAI GSM8K](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c) 完整 test，1,319 条。
- `evaluation/math.jsonl`：与 [lm-evaluation-harness MATH 任务](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hendrycks_math/hendrycks_math_algebra.yaml) 一致来源的完整 MATH test，5,000 条，覆盖七个科目。

保存了完整解答与提取出的最终答案。这里只准备答案数据，尚未实现或运行数学等价评分器。

## 字段和清单

训练 JSONL 主要字段：`sample_id`、`client_id`、`source_dataset`、`source_split`、`source_file`、`source_row`、`input_ids`（文本）、`label`（完整目标文本）。数学另有 `source_family`、`augmentation_type`、`original_question`、`source_problem_id` 等；常识另有 `choices`、`choice_labels`、`answer_index`。

每个目录均有 `client_mapping.json`、`manifest.json`、`quality_report.json`、README。完整版另有 `excluded.jsonl`；抽样版另有 `selection.json`。清单包含每个产物的 SHA-256；下载清单为 [data/reasoning_downloads.json](data/reasoning_downloads.json)。

## 重现与校验

在项目根目录执行：

```powershell
python -X utf8 -u tools/download_reasoning_data.py
conda activate wzm
python -X utf8 -u tools/prepare_reasoning_data.py
python -X utf8 -u tools/validate_reasoning_data.py
```

下载固定上游版本、支持已校验缓存与分段重试。预处理只需额外的 Parquet 读取依赖 `pyarrow`，已存在于当前 wzm 环境。预处理不会覆盖已有已变更的数据目录；再次执行会先验证现有产物。

校验结果：[dataset/reasoning_validation.json](dataset/reasoning_validation.json)。检查包括来源文本逐条一致、排除记录守恒、sample_id 唯一、客户端一致性、抽样 seed 可复现、抽样版本逐条属于完整版本、评测文件保持一致及所有产物哈希。另有 7 项离线单元测试覆盖分段 JSON 读取、可变选项数、布尔标签、数学答案提取和抽样。

## 后续接入边界

新增常识均衡 1000 条版本，已接入原 `main.py`、算法模块、Trainer 和 DataLoader，并提供 harness 评测与断点恢复，见 [README.commonsense-training.md](README.commonsense-training.md)。完整集及 4000 条版本仍保留。旧 zsRE 配置继续走原分支；MetaMath 训练/评测尚未接入，数学的长推理不能沿用通用 512-token 编码静默截断。
训练接入时应沿用现有模型和基本超参，单独处理必要的数据编码、EOS 与评测适配；现有 `step=0` 不表示不限步数。当前文件只保存完整文本，不以截断方式掩盖这类问题。
精确去重/重叠检查不能证明没有语义近似重复；未逐题人工核验上游数学推理正确性。
