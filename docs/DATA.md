# 三套数据集的数据准备

本仓库只包含脚本和配置。原始数据、处理数据、清单产物、模型及历史结果在本地准备，不提交到 Git。

## Commonsense170K 与 MetaMathQA

原始资源存放在 `data/`，处理后存放在 `dataset/`。下载脚本固定来源版本并校验字节摘要，处理脚本记录清洗、客户端映射、抽样和产物摘要。

```bash
python -X utf8 -u tools/download_reasoning_data.py
python -X utf8 -u tools/prepare_reasoning_data.py
python -X utf8 -u tools/validate_reasoning_data.py
python -X utf8 -u tools/prepare_reasoning_data.py --experiment commonsense170k --cap 1000
python -X utf8 -u tools/validate_reasoning_data.py --experiment commonsense170k --cap 1000
```

前三条准备两种任务的完整清洗版和每客户端至多 4000 条版本；后两条补充常识每客户端 1000 条版本。不足抽样上限时保留全部，不重复补样。

| 任务 | 默认训练目录 | 客户端 |
|---|---|---|
| Commonsense170K | `dataset/commonsense170k_client1000_seed42/` | BoolQ、PIQA、SocialIQA、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge、OpenBookQA 各一个 |
| MetaMathQA | `dataset/metamathqa_client4000_seed42/` | GSM/MATH 的 AnsAug、Rephrased、FOBAR、SV 共八种类型 |

Commonsense170K 来源为 LLM-Adapters 作者的数据；MetaMathQA 来源为 `meta-math/MetaMathQA`。精确版本、下载地址和摘要在 `tools/download_reasoning_data.py` 的 PINS 与下载清单中记录，不依赖浮动主分支。

常识的 `test/<client>.jsonl` 保留作者的生成式指令、答案词与选项标签，供主生成评估；`benchmarks/<domain>.jsonl` 保留原生题目字段，供补充 lm-eval 选择题评估。前五个来源采用有标签的 validation，ARC-Easy、ARC-Challenge、OpenBookQA 采用 test；未将无标签测试集当作可评分数据。

数学 `evaluation/gsm8k.jsonl` 与 `evaluation/math.jsonl` 分别保存完整 GSM8K test 和 MATH test。训练保留完整推理和最终答案。数据准备会检查重复、答案冲突、来源映射及与测试题的精确重叠；精确匹配检查不代表不存在语义近似重复。

`data/lm_eval_tasks/` 是可重新下载的固定版本参考资源，不随代码提交；评估运行库由 `tools/install_commonsense_eval.py` 安装到 `.runtime/lm_eval/`。

## zsRE

处理脚本需要 `requirements/requirements-zsre.txt` 中的 embedding/聚类依赖：

```bash
python -m pip install -r requirements/requirements-zsre.txt
python -X utf8 dataset/download_zsre.py --output_dir dataset/zsre/raw
python -X utf8 dataset/process_zsre.py --raw_dir dataset/zsre/raw --output_dir dataset/zsre --num_clients 8 --samples_per_client 500 --seed 42
python -X utf8 dataset/validate_zsre.py --data_dir dataset/zsre --num_clients 8
python -X utf8 dataset/build_zsre_pooled.py --data_dir dataset/zsre --num_clients 8 --seed 42
```

zsRE 保留旧脚本的 `dataset/zsre/raw/` 原始文件位置。默认用固定 seed 的语义聚类分配八个客户端；规模不足时按共同可用数量处理，不重复补样。训练使用 `src → alt`，测试使用 `rephrase → alt`；原始旧答案仅保留为元数据。

严格复现已有划分时应另行保留原数据文件及摘要，并固定 embedding 模型版本和预处理环境。单独重跑脚本不保证与未知版本环境的聚类逐条相同。

## 训练前检查

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedca --check_only
python -X utf8 -u main.py --config config/experiments/zsre.yaml --alg fedit --check_only
python -X utf8 -u main.py --config config/experiments/metamathqa.yaml --alg fedit --check_only
```

`--check_only` 检查本地资源、编码长度、客户端及评估依赖，不加载训练模型。编码拒绝静默截断；长推理样本需要足够的 `max_length` 和显存预算。`--dataset` 可切换同 schema 的完整集或其他抽样版本。
