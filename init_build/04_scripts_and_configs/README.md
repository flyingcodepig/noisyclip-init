# 脚本与配置使用说明

本目录提供部署后开展实验所需的配置模板、预检、顺序调度、环境快照、结果汇总和提交校验脚本。具体模型训练代码由后续实现代理按照 `01_code_architecture` 完成。

## 目录

```text
configs/
  base.yaml
  paths.example.yaml
  schema/experiment.schema.json
  server/l40s_48g.yaml
  server/rtx4090_24g.yaml
  experiments/b0...b6.yaml
  experiments/u1...u6.yaml
scripts/
  config_tools.py
  validate_config.py
  preflight.py
  make_run_id.py
  compare_configs.py
  snapshot_environment.py
  run_experiment.sh
  run_ablation_queue.sh
  collect_results.py
  validate_submission.py
  check_run_artifacts.py
templates/
  experiment_registry.csv
  run_manifest.template.json
```

## 配置继承

实验 YAML 使用 `inherits` 指向父配置。加载器必须：

1. 相对当前 YAML 解析父路径；
2. 递归深合并 mapping；
3. list 整体替换，不做隐式拼接；
4. 检测继承环；
5. 解析 `${oc.env:VARIABLE}`；
6. 最终通过 JSON Schema 和代码级跨字段检查；
7. 保存完整 `resolved_config.yaml`。

例如 B4 继承 B3，因此只写 ELR 的变化。执行配置差异检查：

```bash
python init_build/04_scripts_and_configs/scripts/compare_configs.py \
  --base init_build/04_scripts_and_configs/configs/experiments/b3_trust_weight.yaml \
  --candidate init_build/04_scripts_and_configs/configs/experiments/b4_elr.yaml
```

## 标准启动

容器内：

```bash
export NOISYCLIP_TRAIN_ROOT=/data/train
export NOISYCLIP_TEST_ROOT=/data/test
export NOISYCLIP_RUN_ROOT=/runs

bash init_build/04_scripts_and_configs/scripts/run_experiment.sh \
  init_build/04_scripts_and_configs/configs/experiments/b0_frozen_linear.yaml \
  0
```

批量消融严格顺序执行，避免同一 GPU 争抢：

```bash
bash init_build/04_scripts_and_configs/scripts/run_ablation_queue.sh 0 \
  init_build/04_scripts_and_configs/configs/experiments/b0_frozen_linear.yaml \
  init_build/04_scripts_and_configs/configs/experiments/b1_cosine_proto.yaml
```

## 注意

- 配置模板中的超参数是可靠起点，不是最终最优值。
- `paths.example.yaml` 只用于解释，不要把服务器私有绝对路径提交 Git。
- U 系列必须在对应主线模块完成单种子和复验后运行。
- `u4_balance.yaml` 只有数据审计确认初赛分布不均衡时才启用。
- 所有脚本默认拒绝覆盖已有运行目录或结果文件。

