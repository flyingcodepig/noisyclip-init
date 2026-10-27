# 500类训练与本地实时监控

数据固定500类。此入口负责执行现有B1方案，不代表该方案会优于71分baseline；已完成的B1得分为65。本次新增脚本没有启动新训练。

## 使用

本机双击仓库根目录 `Train-B1.cmd`，输入服务器SSH密码即可。也可以在仓库根目录运行：

```powershell
.\Train-B1.cmd
```

默认创建唯一运行编号，服务器先拉取已推送代码，检查GPU空闲，准备缺失缓存，复用有效冒烟门控，再训练4096步、阶段二1536→2048步，最后推理与验收。源码/配置/质量变化导致门控失效时，才执行一次最终配置完整冒烟。不会再额外测试96×2备选，也不会重新解压已有数据。输入缺失或不一致会停止并显示原因。

本地显示当前阶段、optimizer步数/目标、loss、每步时间和显存峰值；缓存构建及推理显示各自进度。阶段的完整原始日志保存服务器。这里的实时显示频率取决于任务日志，训练通常每5步刷新，缓存每10000张刷新；不是每秒重新SSH登录。

关闭窗口/Ctrl+C只断开监控，tmux中的任务继续。保留窗口开始打印的RunId。

```powershell
# 只查看已有运行，不启动训练，不拉取代码
.\experiments\b1\deployment\Start-B1.ps1 -MonitorOnly -RunId b1_YYYYMMDD_HHMMSS
# 原任务因故失败且已定位修复后，恢复同一任务
.\experiments\b1\deployment\Start-B1.ps1 -Resume -RunId b1_YYYYMMDD_HHMMSS
# 新训练仅训练，不推理
.\experiments\b1\deployment\Start-B1.ps1 -SkipPrediction
```

`Resume`不会跳过checkpoint兼容性校验；如果推理已有部分输出或缓存存在`.building.npy`，现有脚本可能停止，需先检查日志，禁止盲目删产物或伪造通过状态。成功结束的正式任务不需要Resume。

## 文件位置

服务器仓库 `/mnt/noisyclip-b1-20261027`，解释器 `/root/b1-env-20261027/bin/python`。

在 `experiments/b1/code/runs/` 下：

- `<RunId>_pipeline/status.json`：当前阶段及状态。
- `<RunId>_pipeline/progress.log`：本地显示的简明进度。
- `<RunId>_pipeline/*.log`：各步骤完整日志。
- `<RunId>.launch.log`：后台启动错误与异常堆栈。
- `<RunId>_stage1/`、`<RunId>_stage2/`：正式权重与指标。
- `<RunId>_predictions/`：预测CSV/ZIP。

清理后训练缓存约132GiB、测试缓存约73GiB需要重新生成。脚本不会删除数据、升级环境、自动关机或提交比赛；不保存密码。必须先在本机修改并提交/推送代码；脚本检查服务器提交与本机HEAD一致，避免误跑旧版本。500类索引、质量文件及官方特征沿用现有来源绑定。
