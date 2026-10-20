# 联邦方法接入状态（2026-10-20 更新）

已完成公共训练、续训与评估流程的重构，详细命令和实现边界见 [统一训练说明](docs/README.shared-training.md)。

原先“六种方法尚未接入 Commonsense”的检查结论已被本次修改更新。

| 方法 | 算法文件 | 共享流程 |
|---|---|---|
| FFA-LoRA | alg/ffalora.py | 完整保存固定 A 和训练 B |
| FedSVD | alg/fedsvd.py | float32 小核心 SVD、只上传 B、共享续训 |
| FedEx-LoRA | alg/fedexlora.py | 保存与恢复 LoRA 和回写后的基础层 |
| FedRot-LoRA | alg/fedrotlora.py | lam 校验、训练指标、共享续训 |
| FlexLoRA | alg/flexlora.py | 复用聚合、同质 rank、共享续训 |
| FedMomentum | alg/fedmomentum.py | 现有随机化 SVD/残差实现的完整状态和 RNG 恢复 |

上述方法与 FedCA、FedIT、集中式、Local-only 共用 main.py、Trainer、ExperimentRun。
zsRE、Commonsense170K、MetaMathQA 仅提供各自的数据与评分适配。

这表示代码已接通并进行诊断验证，不表示已经完成各方法的 30 轮正式实验。
名为 fedmomentum 的现有算法仍是随机化 SVD 加残差回写，未改成经典服务器动量；与目标论文的对应关系需单独明确。
