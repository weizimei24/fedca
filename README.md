# FedLLM：zsRE、Commonsense170K 与 MetaMathQA

基于 FedLLM Factory 的 LoRA 联邦训练代码，保留三套数据集的数据准备、训练、断点恢复与评估入口。此仓库只分发源代码、配置、依赖和文档；数据、模型、权重、日志及既有实验结果均不随仓库提供。

## 当前训练入口

在项目根目录运行，并始终显式提供 `--config`：

```bash
python -X utf8 -u main.py --config config/experiments/zsre.yaml --alg fedit
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedca
python -X utf8 -u main.py --config config/experiments/metamathqa.yaml --alg fedit
```

统一流程支持：`fedit`、`centralized`、`localonly`、`fedca`、`ffalora`、`fedsvd`、`fedexlora`、`fedrotlora`、`flexlora`、`fedmomentum`。默认训练 30 轮，每 5 轮评估，最后一轮也评估；通过 `--rnd` 和 `--test_gap` 覆盖。方法、协议和恢复机制见 [方法说明](docs/METHODS.md)，数据来源与处理见 [数据说明](docs/DATA.md)。这三份是当前使用文档。

三种任务共用 `main.py → alg.Server/Client → utils.train_utils.Trainer`。公共生命周期位于 `utils/experiment.py`，数据和评估差异分别位于 `utils/dataset_adapters.py`、`utils/evaluation_adapters.py`。

## 环境与本地资源

在适合本机 CUDA/PyTorch 的 Python 环境内安装基础依赖：

```bash
python -m pip install -r requirements/requirements-stage1.txt
python -X utf8 -u tools/install_commonsense_eval.py
```

第二条命令供常识和数学评估使用，按固定版本安装 harness 到本地 `.runtime/lm_eval/`；该目录不提交。zsRE 语义聚类的数据处理另需：

```bash
python -m pip install -r requirements/requirements-zsre.txt
```

模型须另行准备；默认读取本地 `Qwen3-1.7B/`，也可以用 `--model <本地模型目录>` 指定。模型目录应包含权重、模型配置、分词器及 chat template 等所需文件。

## 数据准备

Commonsense170K 和 MetaMathQA 的原始数据下载到 `data/`，处理数据生成到 `dataset/`：

```bash
python -X utf8 -u tools/download_reasoning_data.py
python -X utf8 -u tools/prepare_reasoning_data.py
python -X utf8 -u tools/validate_reasoning_data.py
python -X utf8 -u tools/prepare_reasoning_data.py --experiment commonsense170k --cap 1000
python -X utf8 -u tools/validate_reasoning_data.py --experiment commonsense170k --cap 1000
```

默认处理命令生成完整集及每客户端至多 4000 条版本；后两条补充当前常识训练默认使用的每客户端 1000 条均衡版本。数据来源、客户端映射和清洗规则见 [数据说明](docs/DATA.md)。

zsRE 的下载、处理、校验及集中式数据池：

```bash
python -X utf8 dataset/download_zsre.py --output_dir dataset/zsre/raw
python -X utf8 dataset/process_zsre.py --raw_dir dataset/zsre/raw --output_dir dataset/zsre --num_clients 8 --samples_per_client 500 --seed 42
python -X utf8 dataset/validate_zsre.py --data_dir dataset/zsre --num_clients 8
python -X utf8 dataset/build_zsre_pooled.py --data_dir dataset/zsre --num_clients 8 --seed 42
```

zsRE 沿用现有脚本的 `dataset/zsre/raw/` 原始文件位置。实际客户端规模以预处理记录为准；不足目标规模时不重复补样。重新生成语义划分时，请固定预处理环境及 embedding 模型版本，严格复现既有划分需另行保留原始文件及其清单。

## 检查、续训与评估

```bash
# 检查本地数据、模型、编码及评估依赖，不启动训练。
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedca --check_only

# 训练；中断后使用相同命令和输出目录续跑。
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedca --rnd 30 --test_gap 5

# 对已经保存的轮次补评估。
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedca --rnd 30 --eval_only --eval_rounds 5,10,15
```

自动输出到 `exp/shared/<task>/<dataset>/<alg>/`，包括训练日志、断点、轮次快照和评估结果。新实验可通过 `--suffix` 指定不同目录。数据、模型或训练代码不一致时，断点校验会拒绝续跑。

常识主评估使用与训练相同的 chat 提示，生成答案词后匹配，记录八个来源的 `generation_accuracy` 宏平均；同轮补充原生 lm-eval 0-shot 选择题准确率。`--commonsense_max_new_tokens 32` 控制答案生成预算。

zsRE 使用短答案生成及 EM/F1/containment；数学使用 GSM8K 数值比较及 MATH 的固定 harness 字符串规范化比较，并接受 `The answer is: ...` 格式。数学评估提示不额外要求 boxed，与训练 user 内容保持一致。详细协议和 Local-only 范围见方法说明。

2026-10-21 更新会改变共享训练源码身份及评估协议。旧 shared 实验继续使用原版本续训，新实验使用新 suffix；新生成准确率不能与旧原生选择题分数直接混比。

FedEx-LoRA/FedMomentum 的中途断点引用初始/轮次基础层快照，迁移断点需同时保留 `resume/`、`initial/` 和 `round_*/`。共享身份不再绑定 GPU 编号，但仍校验模型内容、数据、训练设置、代码和关键包版本。

## 旧入口

`config/config.zsre_*.yaml`、`config/config.commonsense1000_*.yaml` 和旧 zsRE 批处理脚本作为兼容入口保留。`rolora`、`flora`、`ilora`、`fedfh` 仅保留原有 zsRE 路线，不在上述统一流程的十种方法内。旧配置的轮数和评估间隔可能不同；新实验优先使用 `config/experiments/`。

旧说明中的 `--config config.zsre_*.yaml` 路径应写成 `--config config/config.zsre_*.yaml`，`requirements-zsre.txt` 应写成 `requirements/requirements-zsre.txt`。旧文档中的实验产物路径只表示运行时输出位置，并不表示仓库附带这些文件。

新仓库根 `.gitignore` 排除所有实验输出；它优先于旧文档中“保留日志/指标”的历史描述。根目录未分发旧的通用 `config.yaml`，因此每次运行必须显式给出任务配置。

## 测试

```bash
python -X utf8 -m unittest discover -s tests -v
```

大部分回归使用随机微型模型或合成数据，不下载真实权重。依赖本地 Qwen tokenizer、Commonsense 数据的集成测试在资源缺失时跳过；真实 harness 评分需要先安装固定评估依赖。完整恢复测试不代表已经完成正式 30 轮质量实验。
