# 方法、运行流程与评估协议

本项目基于 FedLLM Factory。日常使用说明为根 README、本文件及 [数据说明](DATA.md)；旧说明和兼容入口用于复现以前的运行方式。

## 一条公共主流程

```text
main.FedSim
  utils.experiment.prepare_experiment
    utils.dataset_adapters 中的任务子类
  alg.<method>.Server / Client
    utils.train_utils.Trainer
  utils.experiment.ExperimentRun
    utils.evaluation_adapters
```

数据适配器负责提示、编码、客户端和评估集；算法负责本地更新及聚合；运行器负责提交断点、恢复、评估和缓存。公共基础能力位于 `utils/runtime_utils.py`，不包含数据集领域常量。

统一流程支持下列方法，默认 30 轮、每 5 轮评估并评估最后一轮，所有客户端参与每一轮。

| 方法参数 | 实现及状态 |
|---|---|
| `fedit` | 按客户端数据量平均 A、B 因子 |
| `centralized` | 合并全部客户端训练数据，使用一个客户端 |
| `localonly` | 客户端独立训练和保存适配器，不聚合 |
| `fedca` | 本地非真值 KL 约束、训练子集上的候选步长评分及吸收 |
| `ffalora` | 固定 A，只训练和上传 B；完整保存 A、B |
| `fedsvd` | 固定 A 的本地更新和小核心 SVD 重分解 |
| `fedexlora` | 完整保存 LoRA 及聚合回写后的基础层 |
| `fedrotlora` | LoRA 对齐与聚合，`lam` 控制对齐混合 |
| `flexlora` | 乘积空间的小核心 SVD；当前使用同质 rank |
| `fedmomentum` | 低秩聚合的主成分留在 LoRA、能量阈值内残差写回基础层 |

`fedmomentum` 是本仓库的残差回写实现，不是经典 FedAvgM。2026-10-21 改用与 FlexLoRA 共用的 QR 小核心精确 SVD，去掉稠密更新和随机 sketch。它会改变浮点路径和 RNG 消耗，旧版本实验须使用原代码继续；不能将两种实现的实验当作完全相同的运行。

## FedCA

客户端使用下载的全局适配器作冻结教师，仅对答案位置计算 CE 和移除真值 token 后的 KL；教师、学生均关闭训练缓存。动态 padding 使用 tokenizer 的真实 pad ID。

服务器在客户端固定训练子集上比较候选吸收步长，测试集不参与步长选择。关闭约束并固定步长为 1，可退化为 FedIT。常见设置：

```text
完整 FedCA       --ca_lambda 1 --ca_eta_mode probe
仅探测消融      --ca_lambda 0 --ca_eta_mode probe
仅约束消融      --ca_lambda 1 --ca_eta_mode fixed --ca_eta 1
服务器步长 2    --ca_lambda 0 --ca_eta_mode fixed --ca_eta 2
FedExP          --ca_lambda 0 --ca_eta_mode fedexp
```

其余约束、候选网格和客户端权重参数见 `alg/fedca.py`。训练检查点保存 CE/KL 统计、客户端报告和 RNG。

## 三种任务的评估

**Commonsense170K：** 主协议为 `commonsense_chat_generation_plus_native_v2`。生成提示与训练相同，读取 `test/<client>.jsonl`，非 thinking chat、greedy、默认最多生成 32 token。提取 `true/false`、`solution1`、`answer1` 等该题允许的答案词，以完整 token 边界匹配；缺失或歧义答案判错。主要指标为八个来源的 `generation_accuracy` 宏平均，每个来源等权。

同一轮还读取 `benchmarks/<domain>.jsonl`，运行固定 `lm_eval==0.4.13` 的原生 0-shot 选择题评估，不包 chat，记录补充指标 `native_mc_raw_accuracy`。两者有不同任务名、数据文件、协议和指标，不能把生成准确率与旧的原生选择题分数直接混比。可用 `--commonsense_eval native` 将原生指标设为主指标，两种评估仍保留。`--commonsense_max_new_tokens` 单独控制生成答案的预算。

**zsRE：** 保留非 thinking 短答案 chat、greedy 和默认 32-token 预算，记录 EM/F1/containment。共享结果为 0–1 比例；旧独立 `eval_zsre.py` 的相关分数仍为百分数。

**MetaMathQA：** 协议为 `math_chat_greedy_training_prompt_v3`。评估 user 内容与训练一致，不额外添加 boxed 指令；接受 boxed 和 `The answer is: ...` 等明确结尾。GSM8K 进行精确数值/分数比较，MATH 使用固定 harness 的字符串规范化比较。该评分不提供任意符号等价证明。主指标为 GSM8K 和 MATH exact match 的宏平均，同时保存各自分数。

Local-only 的常识和 zsRE 模型评估对应的来源/客户端测试集；数学每个客户端评估 GSM8K 和 MATH。它与全局模型覆盖全部任务的范围不同，结果保留客户端标识。

## 断点与实验版本

优化器边界保存参数、优化器、scaler、数据游标、统计量和 RNG；完成客户端后提交其更新。双槽文件和原子指针避免半写入断点。

FedEx-LoRA/FedMomentum 的基础层在初始状态和轮次快照中保存，中途断点只保存引用及 SHA-256。恢复时同时校验引用文件的身份与内容；不能只搬 `resume/` 而漏掉 `initial/` 和 `round_*/state.pt`。缺失或损坏时明确报错。

共享训练身份不再包含 GPU 编号；数据、模型内容、训练参数、源码和关键包版本仍必须匹配。换 GPU 编号可续训，但不同硬件/软件之间不保证逐位相同。

legacy 的当前源码和依赖使用 `source_sha256`（UTF-8、LF 规范化）校验，历史原始字节摘要仅作为显式兼容别名保留。不同内容不因换行兼容而被放行。

本次修改会改变共享源码身份。已经运行的 shared 实验继续使用其原版本；新实验使用新 suffix。旧 `commonsense: true` 和 zsRE 配置保留，待在跑实验结束、额外旧方法迁移后再删除；不把旧断点直接转换为新格式。原仓库的 realistic/event 模式不属于此次三个任务的公共支持范围。
