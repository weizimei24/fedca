# 统一的训练、断点恢复与评估

2026-10-20 复查修复及验证结果见 [代码复查报告](REVIEW.shared-runtime.md)。本版加强了模型/环境指纹；已有 shared 实验保留原版本续跑，使用本版进行新实验时请用新 suffix。legacy 断点仍走精确兼容表。

## 运行结构

继续使用原来的 `main.FedSim → alg.Server/Client → utils.train_utils.Trainer`。

- `utils/experiment.py`：所有已注册任务共用的运行生命周期、客户端提交、轮次快照、评估调度和缓存。
- `utils/runtime_utils.py`：原子保存、双槽断点、随机数、动态 padding 等公共基础能力。
- `utils/dataset_adapters.py`：数据校验、编码、客户端数据和评估任务定义；不实现续训。
- `utils/evaluation_adapters.py`：常识、zsRE、数学各自的预测和评分；不负责加载算法权重。
- `utils/method_state.py`：方法能力及完整状态。`Server.export_state()` / `restore_state()` 是运行器调用的接口。

固定的 LoRA A 也是模型状态。修改过的基础层必须和适配器一起保存。优化器断点仍只保存它需要的可训练参数与优化器状态；评估完整性不再由 `requires_grad` 决定。

## 统一配置

`config/experiments/defaults.yaml` 提供公共训练设置，各任务配置通过 `extends` 继承，只改数据及协议需要的参数：

```text
config/experiments/commonsense170k.yaml
config/experiments/zsre.yaml
config/experiments/metamathqa.yaml
```

三者都使用 `runtime: shared`。支持：

```text
fedit centralized localonly fedca
ffalora fedsvd fedexlora fedrotlora flexlora fedmomentum
```

在项目根目录、正确的 Python 环境中执行。例如：

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedca
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg ffalora
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedsvd
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedexlora
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedrotlora
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg flexlora
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedmomentum
```

参照方法：

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedit
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg centralized
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg localonly
```

集中式自动合并所有客户端训练行，推导为一个训练客户端；其他方法从数据推导客户端数，无须另写集中式数据加载和配置。若显式给 `--cn`，会检查其是否正确。

zsRE 和数学仅更换任务配置，方法参数保持一致：

```bash
python -X utf8 -u main.py --config config/experiments/zsre.yaml --alg fedca
python -X utf8 -u main.py --config config/experiments/metamathqa.yaml --alg fedca
```

默认 30 轮、每 5 轮评估；最后一轮也评估。统一覆盖：

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg flexlora --rnd 30 --test_gap 5
```

`step: 0` 表示完成配置的本地 epoch。对 1000 条/客户端、bs=1、grad_accum=8、epoch=1，是每客户端 125 次更新；集中式 8000 条是每轮 1000 次更新。显式给较小的 `--step` 会限制每个客户端的更新数，并把产物标记为诊断预算。

默认数据为：Commonsense 每客户端 1000 条、zsRE 既有划分、MetaMathQA 每客户端 4000 条。公共适配器也支持同一 schema 的完整集/其他已处理抽样版，按实际文件推导样本数；用 `--dataset <目录名>` 切换。训练拒绝静默截断；遇到超长样本会报告长度，由实验配置决定长度预算。

## 检查、续训与补评估

先检查数据、编码、客户端、模型指纹及评估依赖，不分配训练模型：

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedexlora --check_only
```

断点默认每 25 个优化器更新保存，并在客户端和轮次完成时提交。中断后重新执行相同命令，复用已经完成的客户端；同样适用于上述所有任务和方法：

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedexlora
```

只评估已经完成的指定轮次：

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedexlora --eval_only --eval_rounds 5,10,15
```

评估失败发生在轮次提交之后；重启补评估不重训该轮。评估历史轮次会恢复那一轮的完整算法状态，结束后恢复当前训练的基础层、适配器和随机数。随机化聚合也恢复正确的聚合前 RNG。

`--eval_limit 1` 仅用于接口检查，和正式全量评估使用独立缓存，不能当作正式实验结果。

## 输出与日志

默认自动隔离到：

```text
exp/shared/<task>/<dataset>/<alg>/
  train.log
  run_manifest.json
  resume/latest.json
  resume/state_0.pt / state_1.pt
  round_001/state.pt
  round_001/complete.json
  round_001/training_metrics.json
  evaluation/round_005/<evaluation-id>/...
  evaluation/round_005/latest_full.json
  progress.json
  completed.json
```

用 `--suffix shared/my_experiment/fedexlora` 自定义输出。更改训练设置进行新实验时使用新的 suffix；断点会拒绝数据、模型、方法参数或训练代码不一致。纯 Python 源码指纹规范化 CRLF/LF，减少 Windows/Linux 检出行尾差异；数据和权重仍按原始字节校验。

`train.log` 保存 logging 日志。若需要所有 print、进度条、异常在内的完整终端输出，在 Linux 使用：

```bash
set -o pipefail
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedca 2>&1 | tee -a commonsense_fedca_full.log
```

Git 忽略共享输出目录中的模型和优化器权重，保留配置、日志、指标和状态清单。

## 评估协议

- Commonsense：与原实验一致，真实 `lm_eval==0.4.13` 引擎，8 个原生选择题任务，0-shot、不额外包 chat prompt，以 raw accuracy 宏平均为主要指标。
- zsRE：沿用原来的非 thinking 短答案提示、greedy 生成、32-token 上限和 EM/F1/containment；新训练使用动态 padding。共享结果中的主要指标统一为 0–1 的比例，旧 `eval_zsre.py` 输出仍为百分数，不能直接混淆数值尺度。
- MetaMathQA：保留完整推理和 EOS 训练。评估采用非 thinking chat、greedy 生成及明确的 boxed 最终答案提示；GSM8K 进行数值答案比较；MATH 使用固定 harness 版本的字符串规范化比较，不声称提供任意符号等价证明，也不声称与上游所有 few-shot 数学协议完全相同。该协议单独记录和缓存。
- Local-only：zsRE/常识每个独立客户端评估其对应组/来源；数学每个独立客户端都评估 GSM8K 和 MATH，结果保留客户端标识。

基础训练环境安装见 `requirements/requirements-stage1.txt`。常识和数学的固定评估依赖：

```bash
python -X utf8 -u tools/install_commonsense_eval.py
```

该脚本读取 `requirements/requirements-commonsense-eval.txt`，将固定依赖安装到 `.runtime/lm_eval/`。

## 旧实验兼容

原有 `config/config.commonsense1000_*.yaml`、`config/config.zsre_*.yaml` 保留 legacy 路线；不会自动将旧断点当成共享断点。新增任务和方法都使用上面的公共配置，不再添加数据集专属恢复类。

对于本次重构前的 Commonsense 断点，`utils/legacy_shared_compat.json` 记录了明确的源码前后哈希和新增依赖哈希。仅在配置、数据、模型及包版本原有校验一致，并且源码恰好匹配这些已验证改动时保留原身份；任意未登记改动仍拒绝恢复。迁移旧实验要同时保留该 JSON。旧输出未被改写或重训。

旧 `tools/run_commonsense.py` 现作为公共配置的兼容参数入口，会选择新共享输出；已有 legacy 实验请继续使用原 `main.py --config config/config.commonsense1000_*.yaml` 命令。

## 方法实现边界

本次补齐 FedSVD 的 float32 小核心分解和只上传 B，避免复制整个基础模型；补齐现有 FedMomentum 实现的固定 rank 形状、PEFT 缩放和残差保存。

名为 `fedmomentum` 的现有实现是随机化 SVD 加残差回写，未新增经典 FedAvgM 服务器动量规则。接入公共流程不代表已经验证其与某篇同名论文完全一致；论文比较应另外明确方法定义。FlexLoRA 当前仍使用同质 rank。

## 增加新任务或方法

新任务：登记 DatasetAdapter 的编码与任务定义，并增加评分实现。不要复制 ExperimentRun 或 Trainer。

新方法：登记能力及影响训练的参数；普通 LoRA 方法复用默认状态接口，具有额外服务器状态的算法覆盖 export_state/restore_state 并完整保存该状态。增加中断恢复对照测试后，再把方法加入实验配置。

## 迁移新增文件

除旧迁移清单中的共享代码外，新增必传：

```text
utils/experiment.py
utils/runtime_utils.py
utils/dataset_adapters.py
utils/evaluation_adapters.py
utils/method_state.py
utils/legacy_shared_compat.json
config/experiments/
requirements/
```

原有 main.py、alg、utils、eval_zsre.py 等被修改的文件也应同步更新，避免混用版本。`tests/test_shared_runtime.py` 和本文建议一并保留。
