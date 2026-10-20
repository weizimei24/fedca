# 共享训练框架代码复查（2026-10-20）

范围：项目内 Python 文件的语法检查；zsRE、Commonsense170K、MetaMathQA 的入口、数据加载、训练、十种已注册方法、断点、评估、缓存和旧入口兼容。没有启动正式的 30 轮实验，也没有改写训练数据、基础模型或原实验权重。

## 已修复的问题

| 问题 | 修复与验证 |
|---|---|
| GSM8K 把分数的分母当作最终答案 | 完整提取嵌套 boxed 表达式，以精确有理数比较整数、小数、分数；拒绝零分母和截断的 box |
| zsRE 生成长度固定为 32，忽略共享参数 | 共享入口及独立 eval_zsre.py 都传入实际 max_new_tokens，元数据和截断标记一致；默认仍为 32 |
| zsRE 评估留下 tokenizer 的 padding_side 状态 | 使用 finally 恢复原值 |
| FedMomentum 零/低秩更新后部分 A、B 同时为零，失去学习方向 | 保留未使用分量的 A，B 保持零；零更新的下一次反向传播仍有可学习梯度 |
| Local-only 最后一个客户端完成后中断，恢复多消耗一次抽样 RNG | 与其他方法一致恢复客户端提交后的 RNG；单轮终点和 CUDA 对照验证 |
| progress.json 写入中断后一直缺失/滞后 | 从权威续训状态恢复进度；eval_only 不把原训练目标改成命令默认轮数 |
| 只校验缓存输入，缓存内指标被改动仍可复用 | 增加缓存内容 SHA，篡改/意外修改会明确报错 |
| 模型只检查“有一个权重文件”，缺少分片也通过检查 | 校验索引引用分片，支持嵌套分片和 HF 缓存链接；指纹包含 tokenizer/generation/chat-template 资产 |
| Windows/Linux 数据指纹键使用不同路径分隔符 | 统一为 POSIX 相对路径 |
| YAML 中错误 runtime 可能走错分支，浮点/布尔整数参数检查不足 | 早期拒绝不合法 runtime、续训模式和整数预算参数 |
| 两个旧 zsRE 批量脚本引用根目录已移动配置 | 改为 config/config.zsre_*.yaml |
| 基础环境未声明 harness 的 scikit-learn 依赖 | 在 requirements/requirements-stage1.txt 补充声明；没有修改已安装环境 |

对应生产文件：utils/evaluation_adapters.py、eval_zsre.py、alg/fedmomentum.py、alg/localonly.py、utils/experiment.py、utils/dataset_adapters.py、utils/model_utils.py、utils/options.py、两个旧 zsRE 启动脚本及依赖清单。

## 验证

- 139 个项目 Python 文件 AST 检查通过。
- 全量 unittest discovery：108 项通过。
- 修复后共享流程与边界回归：27 项通过；最后的 CLI/模型分片等补充检查：17 项通过。
- 新增回归文件 tests/test_runtime_review.py。最初的七个问题先复现失败，再修复到通过；Local-only 单轮 RNG 问题另有独立失败复现。
- 十种方法在微型 Qwen 的 CUDA 训练中，对客户端训练中断、最后一个客户端完成后的中断进行恢复；模型状态和 RNG 与连续运行一致。FedRot-LoRA 检查到第二轮，以执行对齐分支。
- 真实 harness 对 8 个任务各一条样本完成评分；zsRE 生成接口完成检查。随机微型模型得分仅用于接口检查。
- 三套真实数据的 --check_only 都通过：Commonsense 8000 条、zsRE 4000 条（集中式入口）、MetaMathQA 32000 条。
- 现有 legacy FedCA 断点的身份匹配检查通过。兼容表只接受明确登记的源码版本及依赖哈希；其他更改仍拒绝恢复。
- git diff --check 通过。

机器可读检查记录：results/data_audits/shared_runtime_review.json。

## 版本与实验边界

数学答案提取和 zsRE 可配置生成协议分别记为 v2，评估实现变化会产生新缓存，不复用旧评分。

本次还补充模型资产、tokenizers/accelerate 和可移植数据路径指纹。因此，本次修复前的 shared 断点与本版源码身份不同，不能混用。已经运行的 shared 实验保留原版本继续运行；采用本版开始新实验时使用新 suffix，例如：

```bash
python -X utf8 -u main.py --config config/experiments/commonsense170k.yaml --alg fedit --suffix shared/review_v2/commonsense/fedit
```

legacy Commonsense 断点继续通过独立的精确兼容表校验，不会被当作 shared 断点迁移或重训。

FedMomentum 仍指仓库现有的随机化 SVD 加残差版本；本次没有把它改成经典 FedAvgM，也不据此宣称与某篇同名论文完全一致。没有为每个方法运行完整 30 轮质量实验，验证结论限于上述代码和运行路径检查。
