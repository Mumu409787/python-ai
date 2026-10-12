# 问题分类模型训练

已完成字符 TF-IDF + 多类别逻辑回归训练，识别认证(auth)、限流(rate_limit)、故障(failure)、其他(other)。它只辅助显示问题主题，不决定回答、检索策略或拒答；分类“未确定”仍可正常问答。

## 环境与重训

训练使用独立 Python 3.12，Web 主服务继续使用 Python 3.14 和 NumPy。主服务读取 JSON 参数，不加载 pickle/sklearn；体验网页不必先安装训练环境。

在 Windows **PowerShell** 逐行输入；不是网页问答框或 Python `>>>`：

```powershell
Set-Location 'D:\Projects\python-ai'
uv sync --project training --locked --python 3.12
New-Item -ItemType Directory -Path '.demo-run' -Force | Out-Null
Copy-Item training/artifacts -Destination (Join-Path '.demo-run' ('classifier-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))) -Recurse
.\training\.venv\Scripts\python.exe training/train_classifier.py
Get-Content training/artifacts/evaluation.json -Encoding utf8
```

备份后再训练，训练会覆盖当前模型和评估。模型含词表、IDF、系数、截距、类别、阈值与数据 SHA-256。主服务按文件修改时间读取新参数。joblib 文件仅留本地，不提交、不由API加载。

| 文件 | 作用 |
| --- | --- |
| data/questions.jsonl | 104 条人工准备的教学问题、主题标签与固定 split/group |
| [data/README.md](data/README.md) | 类别定义及数据划分规则 |
| train_classifier.py | 特征、训练、验证选阈值与测试评估 |
| artifacts/question_classifier.json | 业务推理参数 |
| artifacts/evaluation.json | 指标、混淆矩阵、逐题预测与概率 |

## 指标与已知错误

训练/验证/测试为64/16/24；检查规范化重复与分组跨集合重复。验证阈值0.40，测试原始accuracy、macro-F1均1.00；应用阈值接受22/24（91.67%），另2条未确定，接受项在此测试集全部正确。主服务推理与sklearn保存的24条概率逐项一致。

这是手工教学小样本，词汇/主题相近，不是严格域外泛化测试，不能宣称生产准确率100%。额外探索已发现：

- “如何给数据库事务设置隔离级别”被误判故障，约0.514；
- “请解释令牌桶限流算法”约0.334，显示未确定；
- “CRANE-18743 对应的图像服务端口是多少？”被误判限流，约0.42。

这些探索未用于反向调参美化测试指标。后续加入人工核验真实难例，保留新的独立测试集再比较。分类概率未做校准，也不是答案正确率。

## 单独验收推理

主服务启动后，在 **PowerShell** 输入，不调付费模型、不写会话：

```powershell
$body = @{ message = '访问令牌多久过期' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/chat/classify' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 5
```

预期auth/认证；42901预期限流，毕业答辩问题低置信度。缺失/损坏模型显示unavailable，问答继续；低分显示未确定。环境缺包、下载失败、基础解释器权限问题见 [故障排查](../运维/故障排查.md)。

## LoRA 条件评估，尚未执行微调

本机RTX4060 Laptop显存8188MiB、总内存约15.2GiB，可作为小模型微调试跑起点，但本次没有安装CUDA版PyTorch、下载大模型权重或运行LoRA。分类器训练不等于微调DeepSeek API模型，API使用权也不是模型权重。

继续微调需选择有可用权重及合适许可的模型，另外准备人工核验的指令/回答数据与独立验证/测试集，检查PyTorch CUDA设备识别及Transformers/PEFT/Accelerate/Datasets的兼容组合；QLoRA另需bitsandbytes。先短序列、小批次试跑并记录峰值显存、耗时和验证损失，不能仅凭量化权重体积保证训练可容纳。

Windows有官方bitsandbytes构建，WSL2不是所有路线的强制前提。参考 [PEFT量化训练](https://huggingface.co/docs/peft/developer_guides/quantization)、[bitsandbytes安装](https://huggingface.co/docs/bitsandbytes/main/en/installation)。104条主题标签不是指令问答数据，不能冒充大模型微调集。实际训练完成并与未微调模型比较后，才能报告微调收益。
