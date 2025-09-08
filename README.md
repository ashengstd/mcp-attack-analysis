# MCP Attack Analysis

实验性项目，用于复现、评估与分析「多步骤工具调用攻击」（Model Context Protocol / Toolformer 风格）对接入外部工具的 LLM 带来的安全风险。

## 功能概览

- 📦 **攻击场景数据集**：基于 Hugging Face `load_from_disk` 的离线数据，涵盖 `attack_id`、`server_name`、`tool_name`、`query`、`tool_content`、`is_attack` 等字段。
- 🧪 **评估脚本**：提供批量并发评估 (`eval/parallel.py`) 与单场景多轮回放 (`eval/rollout.py`)，自动抽取模型输出中的工具调用并统计命中率。
- 🧠 **强化学习微调**：`train/rlvr.py` 基于 TRL 的 GRPO (RLVR) 流程，结合自定义奖励 `train/reward.py`，支持对模型进行拒绝攻击意图的反向强化学习训练。
- 🛠️ **实用工具**：`utils/attack.py`、`utils/tool.py` 提供数据类、格式化与解析工具；`update-deps.py` 可将当前环境版本同步到 `pyproject.toml`。

## 仓库结构（节选）

```
├── config/            # Accelerate/Deepspeed/FSDP 配置示例
├── data/attacks/      # Hugging Face parquet 数据集 (train/test)
├── eval/
│   ├── parallel.py    # 全量异步评估入口
│   └── rollout.py     # 单样本多次回放评估
├── train/
│   ├── rlvr.py        # 基于 GRPO 的强化学习训练脚本
│   └── reward.py      # 工具调用奖励函数
├── utils/             # 数据类与工具函数
├── requirements.txt   # 通过 uv 锁定的依赖 (生成自 pyproject.toml)
├── pyproject.toml     # 项目元数据与依赖声明
└── .env.example       # 运行所需的核心环境变量模板
```

## 快速开始

### 1. 准备 Python 环境

- Python 版本：`>= 3.12`
- 推荐使用 [`uv`](https://github.com/astral-sh/uv) 管理依赖（仓库自带 `uv.lock`）

```fish
uv sync
source .venv/bin/activate.fish
```

若未安装 `uv`，可以使用原生 `pip`：

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你使用的 LLM 服务信息：

```
MODEL=your-deployment-or-model-id
API_KEY=xxxxxxxx
API_BASE=https://api.your-provider.com
```

脚本会通过 `python-dotenv` 自动加载 `.env`。如果使用本地模型，可将 `API_BASE` 留空或改为本地推理服务地址。

### 3. 准备数据集

仓库已包含处理好的数据集 (`data/attacks/`)。

- 训练集与测试集使用 Hugging Face `DatasetDict` 保存（Arrow/Parquet）。
- 如需替换为自定义数据，可调用 `datasets.Dataset.save_to_disk()` 导出同名字段目录。

数据字段说明：

| 字段           | 类型   | 说明                              |
| -------------- | ------ | --------------------------------- |
| `attack_id`    | string | 样本唯一标识                      |
| `server_name`  | string | 目标服务/系统名称                 |
| `tool_name`    | string | 暴露给模型的工具名称              |
| `tool_content` | string | 工具描述/诱导文本                 |
| `query`        | string | 攻击者输入模型的自然语言请求      |
| `is_attack`    | bool   | 是否为攻击意图（True 为攻击场景） |

## 评估流程

### 全量并发评估：`eval/parallel.py`

```fish
python eval/parallel.py
```

- 一次性遍历数据集中的所有场景，并发度由 `MAX_CONCURRENT_REQUESTS` 控制（默认 64）。
- 运行结束后会输出总体攻击命中率，并在 `results/results.json` 写入每条样本的响应与判定。

### 单样本多轮回放：`eval/rollout.py`

```fish
python eval/rollout.py
```

- 随机抽取一个攻击场景，重复执行多轮 (`ROLLOUTS` 默认 16) 验证模型是否稳定拒绝攻击。
- 有助于观察模型对同一场景的波动情况。

两份脚本共享以下核心组件：

- `utils.attack.AttackScenario` / `AttackResult`：数据结构定义。
- `utils.llm.LLMClient`：基于 `litellm` 的统一 LLM 调用封装。
- `utils.tool.extract_tool_call`：解析模型输出中的结构化工具调用。

## 强化学习微调 (GRPO / RLVR)

若希望让模型学习“拒绝恶意工具调用”，可以运行：

```fish
python train/rlvr.py
```

- 该脚本使用 TRL 的 GRPO (Grouped Regularized Policy Optimization) 策略，实现 RLVR 流程。
- 奖励函数位于 `train/reward.py`：
  - 攻击场景命中工具调用 → 负奖励
  - 攻击场景拒绝调用 → 正奖励
  - 正常场景误拒绝 → 负奖励
  - 统一要求输出合法 JSON 结构，否则奖励置 0
- 需要提供满足内存的机器（默认启用 Flash Attention、bfloat16、vLLM 推理卸载）。
- `config/fsdp.yaml`、`config/dp.yaml` 演示了加速/分布式训练配置，可自行调整。

## 日志与产出

- `results/`：评估脚本生成的 JSON 结果（默认被 `.gitignore` 排除）。
- `swanlog/`：SwanLab 日志目录，用于追踪训练/评估过程中的指标与样本，可在 SwanLab UI 中可视化。
- 若需要持久化日志，请在运行前自行备份或修改脚本输出路径。

## 实用脚本

- `update-deps.py`：读取当前虚拟环境中的包版本，并写回 `pyproject.toml`。支持 `--exact` 强制使用 `==` 版本拼写，默认写入 `>=`。
- `.python-version`：指明默认的 `pyenv`/`uv` Python 版本（3.12）。

## 安全与伦理注意事项

- 仅将此项目用于安全研究、红队评估或防御性分析。
- 数据集中包含大量可能造成敏感信息泄露或系统破坏的提示示例，请勿在生产系统上尝试。
- 若调用第三方 LLM API，请遵守服务商的使用条款，并注意费用与速率限制。

## 贡献

欢迎通过 issue / PR 贡献：

- 改进工具调用解析与判定逻辑
- 增补更多攻击场景或非攻击样本
- 引入不同的奖励函数或训练范式

提交前请确保新增代码通过必要的构建或测试，并与项目风格保持一致。

## 许可

仓库目前未附带正式开源许可。如需使用或再发布，请先联系作者（pyproject 中的邮件地址）确认授权。若希望采用常见开源许可证（MIT / Apache-2.0 等），可在 issue 中讨论后补充。
