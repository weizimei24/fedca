# 新 GitHub 仓库上传清单：仅三套数据集相关代码

本清单依据 2026-10-20 当前工作区整理，范围是 **zsRE**（此前提到的 zrse/zrses）、**Commonsense170K** 和 **MetaMathQA** 的训练、评估及数据重建代码。

准确逐文件白名单为 [upload-code-only-manifest.json](upload-code-only-manifest.json)。共列出 **108 个来源文件**；加上根 README 和 `.gitignore` 的两份映射，目标仓库为 **110 个文件**。`groups` 中的每一个路径都表示要复制的单个文件，不能据此递归上传它所在的文件夹。下面各组共同组成推荐上传范围。

## 1. 统一训练与评估的必要文件

- 根入口：`main.py`、`eval_zsre.py`。
- 算法：`alg/__init__.py`、`base.py`、`ftbase.py`，以及 `fedit.py`、`centralized.py`、`localonly.py`、`fedca.py`、`ffalora.py`、`fedsvd.py`、`fedexlora.py`、`fedrotlora.py`、`flexlora.py`、`fedmomentum.py`。
- 公共配置：`config/experiments/defaults.yaml`、`zsre.yaml`、`commonsense170k.yaml`、`metamathqa.yaml`。
- `utils/`：只复制 JSON 白名单的 `shared_utilities_and_compatibility_dependencies` 组所列文件。
- 环境：`requirements/` 下所列的三个依赖文件，`tools/install_commonsense_eval.py`、`tools/run_commonsense.py`。

`utils/experiment.py`、`runtime_utils.py`、`method_state.py` 是公共运行和恢复能力；`dataset_adapters.py`、`evaluation_adapters.py` 承接三种数据/评分协议。不能只复制按数据集命名的文件。

以下文件也需要保留：

- `utils/train_utils.py`：原框架的实际训练器。
- `utils/qa_utils.py`：虽有旧任务函数，但 `data_utils.py` 当前直接导入它，不能因文件名不含这三套数据集就删除。
- `utils/sys.yaml`、`utils/eval.yaml`：被公共模块读取的运行配置。
- `utils/commonsense_training.py`、`commonsense_framework.py`、`commonsense_fedca.py`：当前主入口及兼容分支仍有调用；不能直接移除。
- `utils/legacy_shared_compat.json`：源码兼容映射，是代码元数据，不是数据集或实验结果。

新实验使用 `config/experiments/`。这套统一流程支持上面十种方法，默认 30 轮、每 5 轮评估，断点由公共运行器管理。

## 2. 数据下载、处理和校验代码

保留这些脚本，以便新电脑在不上传数据的情况下重新准备资源：

```text
tools/download_reasoning_data.py
tools/prepare_reasoning_data.py
tools/validate_reasoning_data.py
dataset/__init__.py
dataset/download_zsre.py
dataset/process_zsre.py
dataset/build_zsre_pooled.py
dataset/validate_zsre.py
dataset/check_zsre_protocol.py
dataset/smoke_zsre.py
```

**`dataset/` 只上传上述七个 `.py` 文件。** 不上传其中任何数据子目录、JSON/JSONL 数据、样本抽样记录、统计/校验报告或自动生成的数据集 README。`data/` 整个目录不上传；其中 `lm_eval_tasks/` 也属于下载资源，可由脚本重新获取。

## 3. 以前 zsRE 和常识实验的兼容代码

为了保留此前的 zsRE 对比方法、FedCA 消融及旧命令，推荐一并保留白名单的 `legacy_zsre_and_commonsense_entrypoints` 组：

- `alg/rolora.py`、`flora.py`、`ilora.py`、`fedfh.py`，以及 `utils/fh_utils.py`。
- `config/` 中明确列出的 13 个 `config.zsre_*.yaml`、4 个 `config.commonsense1000_*.yaml`。
- `tools/run_fedca_suite.py`、`resume_fedca.py`、`eval_zsre_rounds.py`。
- 根目录的 `run_zsre_sequence.py`、`run_zsre_centralized.py`、`summarize_stage1.py`。

这组包含的是脚本和实验方案，不包含任何已跑出的结果。四种额外算法保留原 zsRE 路线，不代表接入了新的统一十方法运行器。

如果未来明确只保留统一入口，可以另行精简这一组；本次推荐完整保留，以覆盖此前 zsRE 实验。不要单独删掉它再上传全部测试：部分恢复/算法测试会读取这些配置并导入这些算法。

## 4. README、说明及回归测试

保留白名单的 `documentation_and_upload_instructions` 和 `regression_tests` 两组。测试覆盖公共恢复、常识接入、数据准备及相关算法。上传测试源码不需要上传真实训练数据和模型权重；其中部分真实 tokenizer/常识评估测试在运行时仍要求另行准备本地模型、数据及 harness，不能将完整测试集当作无需资源的离线测试。

`docs/REVIEW.shared-runtime.md` 是代码检查说明，可以上传；该文引用的 `results/` 审计产物不上传。数据准备说明中记录的数据规模和来源可以保留，但不打包其引用的数据文件。

新仓库根文件使用以下映射，**不直接沿用当前根 README 和 `.gitignore`**：

| 当前项目中的来源 | 新仓库目标 |
|---|---|
| `docs/README.code-only.md` | `README.md` |
| `docs/code-only.gitignore` | `.gitignore` |

两个来源模板本身也在白名单中，可以同时保留在 `docs/`。

旧 README 中有历史配置路径和阶段性说明，新仓库以新根 README 及 `docs/README.shared-training.md` 为主。现有根 `.gitignore` 为旧实验保留了部分日志、指标和数据，不能用于这次“代码仓库”的筛选。

## 5. 明确不上传

```text
.git/                         旧仓库的全部历史
.env、.env.*                  本地凭据
data/                         全部原始数据、评估数据、下载资源
dataset/ 中未列入白名单的内容  全部实际数据和数据产物
exp/                          所有实验、日志、指标、预测、断点和适配器
results/                      所有统计、审计 JSON、图表等产物
stage1/                       旧实验辅助目录
Qwen3-1.7B/、其他模型目录      模型权重、模型配置、tokenizer 等模型资源
models/、weights/、checkpoints/、adapters/
.runtime/                     下载的评估运行依赖
__pycache__/、.pytest_cache/、缓存和本地环境
.vscode/、.idea/               本地编辑器设置
```

任何位置的 `.pt`、`.pth`、`.bin`、`.safetensors` 等权重/优化器文件均不上传。旧结果中的 `training_config.json`、`metrics.json`、`predictions/*.jsonl`、CSV、日志也不上传，不能因为是文本文件就保留。

未列入白名单的根脚本、其他数据集配置、异步/event 算法、诊断和绘图脚本不上传。例如 `webui.py`、`eval.py`、`eval_zsre_ori.py`、跨域 SQuAD/StackExchange 等流程不属于本次范围。保留代码中的这些可选分支不在本次支持范围，使用新根 README 的三任务命令即可。

`docs/README.migration-three-datasets.md` 是以前“搬到另一台机器”的清单，包含数据等内容，**不用于这次上传**。

## 6. 给另一个窗口的执行说明

可以将下面这段话与新仓库地址一起交给另一个窗口：

> 请按当前项目 `docs/UPLOAD.code-only.md` 和 `docs/upload-code-only-manifest.json` 整理代码，上传到我提供的新 GitHub 仓库。将 JSON 的 `groups` 中全部单文件复制到一个新的空目录，保留相对路径；再按 `root_file_copies` 生成根 README.md 和 .gitignore。只复制白名单，不递归复制任何源目录，不修改原实验目录，不复制原 .git，不复用原提交历史。新目录初始化 Git 后，检查暂存文件恰好等于白名单目标路径加两个根映射文件，确认没有数据、模型、权重、凭据和实验输出，再提交推送。保留相应的上游项目说明；如目标仓库已有文件或历史，先检查后合并，避免覆盖。

新建目录再初始化 Git 是为了让原仓库历史中的数据、结果或权重不会随此次推送进入新仓库；`.gitignore` 不能移除旧历史里已经提交过的内容。清单外文件默认不上传。
