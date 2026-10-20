> 2026-10-20：新增数据集无关的共享训练、断点与评估入口，支持十种方法；新实验请优先阅读 [统一训练说明](README.shared-training.md)。本文原有命令保留为 legacy 实验兼容说明。

# Commonsense170K：FedCA 与三个参照（每客户端 1000 条）

新增数据为 `dataset/commonsense170k_client1000_seed42/`。8 个客户端依次对应 BoolQ、PIQA、SocialIQA、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge、OpenBookQA，每个恰好 1000 条，共 8000 条。使用 seed=42 从清洗后的完整训练集无放回抽样，不重复补样。`selection.json` 记录样本 ID，`manifest.json` 记录文件校验和与来源。此前完整集与 4000 条版本保持独立。

现在直接接入原有训练框架：`main.py → FedSim → alg.fedit / alg.centralized / alg.localonly / alg.fedca → 原 Client/Trainer → DataLoader`。原算法类负责客户端调度与聚合；新增模块负责数据、断点和 harness 评测。`tools/run_commonsense.py` 已改为参数转发入口，内部执行同一个 `main.py`，没有单独的训练循环或优化器实现。

复现抽样及来源校验：

```powershell
conda activate wzm
python -X utf8 -u tools/prepare_reasoning_data.py --experiment commonsense170k --cap 1000
python -X utf8 -u tools/validate_reasoning_data.py --experiment commonsense170k --cap 1000
```

## 训练与评测命令

在项目根目录、现有 `wzm` 环境中运行。隔离评测依赖已在本机安装到 `.runtime/lm_eval/`；换机器时先运行下面的安装命令。它使用固定版本与 `--no-deps`，不升级现有 PyTorch、Transformers、PEFT；这些基础库需由现有训练环境提供。

```powershell
conda activate wzm
python tools/install_commonsense_eval.py
```

以下命令各自包含训练、每 5 轮评测和自动续跑，可以依次执行。运行期间会打印训练进度；评测在第 5、10、15、20、25、30 轮完成训练后执行。

```powershell
# FedCA：原有第二步约束 + 第三步 probe 按打分选择 eta。
python -X utf8 -u main.py --config config/config.commonsense1000_fedca.yaml

# FedIT：8 个客户端全参与，每轮从相同全局适配器开始，按样本数平均 LoRA A/B 因子。
python -X utf8 -u main.py --config config/config.commonsense1000_fedit.yaml

# 集中式：合并同一批 8000 条训练数据，每轮完整遍历一次。
python -X utf8 -u main.py --config config/config.commonsense1000_centralized.yaml

# Local-only：8 个独立适配器，各自在本客户端训练并跨轮保留，不做聚合。
python -X utf8 -u main.py --config config/config.commonsense1000_localonly.yaml
```

四份配置均设置 `rnd: 30`、`test_gap: 5`、`resume: auto`、`checkpoint_steps: 25`。可用原框架参数覆盖，例如 `--rnd 30 --test_gap 5`；`--eval-every` 也是 `test_gap` 的别名。最终轮总会评测，例如 12 轮、间隔 5 时评测 5、10、12。默认输出分别位于 `exp/commonsense170k/client1000_seed42/framework/` 下各方法同名目录。更改训练配置重做实验时，用新的 `--suffix`（相对 `exp/`）。

FedCA 使用同样的 30 轮、评测间隔和训练预算，输出到同级 `framework/fedca/`。其方法参数沿用 zsRE FedCA：`ca_lambda=1`、`ca_tau=1`、`ca_eta_mode=probe`、`ca_eta_grid=1,1.5,2,3,4,6,8`、`ca_kappa=inf`、`ca_beta=0`、`ca_probe_size=32`。仅将 probe 前向批次设为 1，以容纳常识任务的长样本；probe 只读取各客户端自己的训练子集，评测仍使用完整基准。关闭旧的重复快照 `ca_save_every=0`，统一由框架每轮提交断点和适配器。

FedCA 的断点额外保存 CE/KL 累计统计、客户端 probe 报告与 excess，恢复中途客户端或跳过已完成客户端时仍可正确选择 eta。`round_005/fedca_metrics.json` 等记录该轮 eta、权重和 probe 得分，原子覆盖同一轮，避免中断重试造成重复日志。长提示的 KL 只在有监督的答案位置构造分布，数学定义不变。三个原参照已有断点会识别本次仅新增 FedCA 的已知兼容代码版本，保持其训练身份；任意模型、数据、训练参数或未认可代码变化仍被拒绝。

不加载 GPU 模型的训练前检查（校验数据、完整编码、8 个任务及依赖）：

```powershell
python -X utf8 -u main.py --config config/config.commonsense1000_fedit.yaml --check_only
```

## 训练预算与基本参数

| 设置 | FedIT | 集中式 | Local-only |
|---|---|---|---|
| 每轮训练数据 | 每客户端 1000 条，8 客户端 | 合并后的 8000 条 | 每客户端 1000 条，8 客户端 |
| 每轮本地 epoch | 1 | 1 | 1 |
| 每轮优化器更新 | 每客户端 125 次，共 1000 次 | 1000 次 | 每客户端 125 次，共 1000 次 |
| 30 轮总训练样本呈现次数 | 240000 | 240000 | 240000 |
| 适配器 | 1 个全局适配器 | 1 个集中式适配器 | 8 个独立适配器 |
| 评测适配器 | 全局适配器评测全部 8 项 | 集中式适配器评测全部 8 项 | 每个适配器只评测自己的来源 |

沿用本地 Qwen3-1.7B、AdamW、初始学习率 `1e-4`、逐轮乘 `0.99`、batch size=1、梯度累积=8、LoRA rank=8/alpha=32/dropout=0.05、q_proj/v_proj、seed=42、float16。与原 Trainer 一样，每个客户端每轮新建 AdamW；**同一次客户端训练中断恢复时，恢复其原优化器状态**。集中式每轮也新建 AdamW。所有方法使用相同初始化、数据、损失、随机种子规则和评测设置。FedIT 聚合保留现有 `FTBaseServer.aggregate` 的因子平均语义；它与平均各客户端 `B@A` 更新不同。

必要的适配：

- 原来的固定 `step=63` 只能处理约 504 条，不能完整覆盖新的 1000 条。配置中 FedIT/Local-only 设 `step: 125`，集中式设 `step: 1000`，默认恰好完整遍历一轮。Commonsense 的 `step=0` 明确定义为跑完指定 epoch；其它数据集继续保留旧有语义。
- SFT 使用原始 LLM-Adapters 指令及完整目标文本，Qwen 非思考聊天模板，屏蔽提示和 padding 的损失，并监督答案及 EOS。
- PIQA 有 4 条所选样本超过原 512 token，最长 1123。训练长度上限设为 1536，动态填充，不截断或删除这些样本。其他来源所选样本均低于 512。
- 默认开启梯度检查点以降低显存占用；可用 `--no-gradient_checkpointing` 关闭。它属于训练身份的一部分，中途不能切换。只加载一个基础模型，各个客户端轮流使用它。
- 原 `main.py`、数据加载、模型加载、算法基类和 Trainer 增加 `commonsense` 分支，复用原有 DataLoader 随机种子与训练入口。旧 zsRE/FedCA 等配置默认 `commonsense=false`，继续走原分支。单卡直接运行即可，不用 `accelerate launch`。
- Commonsense 分支修正原调度“最后 10 轮每轮测试”的行为，严格按 `test_gap` 及最终轮评测。

可配置参数见 `python main.py --config config/config.commonsense1000_fedit.yaml --help`。修改 `--bs`、`--grad_accum` 时，需同时检查 `--step` 是否足以覆盖指定 epoch；也可以设 `--step 0` 完整遍历。表中的次数对应默认配置。

## lm-evaluation-harness 协议

使用真实的 **lm_eval 0.4.13 HFLM + simple_evaluate**。直接读取准备好的 `benchmarks/*.jsonl`，不需要临时从 Hugging Face 下载数据或运行远程数据集脚本。任务使用规范字段；绝不使用 `test/*.jsonl` 的生成式指令替代标准基准。

- `num_fewshot=0`，原生纯文本提示，不额外套聊天模板，用选项的条件对数似然进行 multiple-choice 测评。
- BoolQ、PIQA、SocialIQA、HellaSwag、WinoGrande 使用有标签的 validation；ARC-Easy、ARC-Challenge、OpenBookQA 使用有标签的 test；共 22419 条。
- 7 项已保存的上游任务提示/预处理来自 `data/lm_eval_tasks/` 的固定版本。SocialIQA 在本地定义为 `Q: {context} {question}\nA:`、三个答案候选、原始准确率；它属于本项目明确固定的补充任务定义。
- WinoGrande 沿用上游的两个不同上下文、共同句子后缀的评分方式；HellaSwag 沿用其大小写与 WikiHow 标记清理；ARC 保留实际选项数和标签。
- 保存每项原生 `acc`，有定义的任务还保存 `acc_norm`。主汇总 `macro_acc` 是 8 项 **原始 acc** 的算术平均，避免混用原始和长度归一化指标。指标在 0–1 之间。
- FedIT/集中式是全局模型的 8 领域评测；Local-only 是各自来源的个性化参照，其宏平均不表示某个单一 Local 模型对全部来源的能力。
- 默认评测 batch size=1、上下文上限 4096。可调 `--eval_batch_size`、`--eval_max_length`，设置写入结果身份。运行中不额外加载第二份基础模型，也不合并/丢弃活动 LoRA。

训练使用作者发布的指令/答案格式，评测使用 harness 原生提示与候选答案，这是此处有意固定的协议；不能把结果直接宣称为原 LLM-Adapters 生成式评测结果的复现。BoolQ 原训练指令与带 passage 的规范评测格式也保持各自上游形式。

上游接口说明：[Python API](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)、[任务定义](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)。

## 中断、恢复和补评测

**中断后重新执行原训练命令即可。** 默认 `--resume auto`。程序先验证训练设置、代码、基础模型、数据清单及软件版本，再加载已提交断点；配置不一致会报错，不会悄悄从头开始。

默认每 **25 个优化器更新**保存一次，也在客户端与整轮结束时保存。修改频率示例：

```powershell
python -X utf8 -u main.py --config config/config.commonsense1000_fedit.yaml --checkpoint_steps 10
```

断点包含：已完成轮数、本轮已完成客户端、FedIT 待聚合更新/Local 独立适配器、当前客户端适配器、AdamW 状态、GradScaler、Python/NumPy/PyTorch/CUDA 随机状态和下一批次位置。保存于完成更新并清空梯度的边界；中途断电或 Ctrl+C 后，最近尚未提交的少量更新需要重算，已提交进度继续复用。默认窗口最多约 25 次更新，即默认 batch/累积下最多约 200 条样本的工作。

`resume/state_0.pt` 与 `state_1.pt` 轮换写入，先完整写入并落盘，再原子更新 `latest.json`（带 SHA-256）。发现已提交文件损坏时明确报错，不能删除断点后假装续跑。运行锁防止同一输出被两个进程同时写入。精确一致测试针对同环境、同配置；跨 GPU/软件版本的位级一致不作假设。

每个完成轮保存独立适配器快照。评测前先提交训练进度；评测报错不会使该轮训练丢失。每个基准分别落盘，重复命令只补缺失评测；完整评测与 `--eval-limit` 诊断结果分别保存并校验身份。

只补测已有轮次，例如 FedIT 已训练到第 10 轮：

```powershell
python -X utf8 -u main.py --config config/config.commonsense1000_fedit.yaml --eval_only --eval_rounds 5,10
```

更换配置文件即可补测集中式或 Local-only。没有快照的轮次会报错。若已完成 30 轮，可省略 `--eval_rounds`，补测全部 6 个轮次。`--rnd` 可以增大以继续更多轮，不能小于已完成轮数；训练身份中的其它参数需保持相同。

`--eval_limit 1` 仅用于检查每项 1 条，输出标记 `diagnostic_only=true`，不能当作正式结果，也不会让后来不带该参数的完整评测被跳过。缩短 `step` 的检查也会在 `completed.json` 中标为诊断预算。

## 文件与验证

每种方法的输出：

- `run_manifest.json`：训练设置、模型/数据/代码指纹与预检查报告。
- `resume/latest.json`、`resume/state_*.pt`：最新可继续训练的状态。
- `round_005/` 等：每轮 `adapters.pt`、`complete.json`、训练指标。Local-only 快照含 8 份 LoRA 参数。
- `evaluation/round_005/<评测身份>/`：每项指标、逐题输出与 `summary.json`；`latest_full.json` 指向正式评测汇总。
- `progress.json`：完成的训练轮数；`completed.json`：所请求训练与评测均结束后生成，包含轮数和诊断标记。
- 原框架的 `.log` 文件：通过原 `get_logger` 继续写日志。原控制台 round 下标从 0 开始；快照及评测目录从 1 开始。

适配器与训练状态仅保存在本地，已添加到 `.gitignore`；指标、配置和代码可另行提交。不要只复制 LoRA 快照来假装恢复 AdamW：真正继续训练应保留整个输出目录。

验证包括来源逐条一致、无放回 1000 条抽样、评测数据一致、训练长度/EOS、现有 FedIT 聚合一致性、真实 FedSim/算法/Trainer 调用链下三种方法的中途及轮次/epoch 边界恢复参数逐位一致、评测间隔、任务评分方向、原子提交与哈希拒绝。可运行：

```powershell
python -m unittest tests.test_commonsense tests.test_prepare_reasoning_data
python -m unittest tests.test_commonsense_fedca.FedCACommonsenseTest
# 可选：实际加载本地 Qwen，3 条训练样本 + 各基准 1 条，仅作诊断。
python -X utf8 -u tools/check_commonsense_runtime.py
```

诊断报告在 `results/data_audits/`。本次代码接入和少量验证不代表三个完整 30 轮实验已运行。

原独立入口生成的断点使用不同训练循环与随机排列，不与这次框架断点混用。新配置使用 `framework/` 子目录，避免覆盖旧输出；已有 1000 条数据与 harness 任务定义直接复用。
