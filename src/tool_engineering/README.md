# Tool Engineering

`tool_engineering` 负责治理 Agent 与 AppWorld 环境之间的工具交互界面，核心目标是在**不破坏 Cache-First L0 静态前缀**、**不重建执行器**的前提下，降低探索与回查摩擦、把运行时错误转化为可恢复信号，并前置完成工具输出的信息投影。

---

## 1. 定位与架构边界

结合现有 AgentLab 架构（持久化 AppWorld Python Shell + Requester 底层事件记录 + `context_engineering` + `execution_assurance`），工具工程收敛为三个高杠杆方向，明确不做重复建设：


| 方向                | 决策         | 职责说明                                                                                                                              |
| :------------------ | :----------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Tool Discovery**  | **重点建设** | 渐进式披露（紧凑搜索签名 → 按需完整 Schema）、分页能力标识与检索降噪，缩短找到正确 API 的路径。                                      |
| **Tool Feedback**   | **重点建设** | 基于 Requester 增量事件生成部分成功反馈（防止批量中断后盲目重跑）与结构化错误自愈提示。                                               |
| **Tool Projection** | **重点建设** | 分离**程序可见**（Python Shell 完整变量与 `ObservationStore` 原文）与**模型可见**（紧凑签名与有界预览），在进入上下文前完成信息治理。 |
| **Tool Execution**  | 复用现状     | 已有持久化 Python Shell（支持程序化编排与批处理）及 Requester 事件流，不另建执行引擎。                                                |
| **Tool Guard**      | 复用现状     | 写操作防重放由`AppWorldTool` 负责，终局业务核验由 `execution_assurance`（`reflection_v1` / `shadow`）负责，不在工具层叠加重型拦截器。 |

### 与其他模块的协作红线

1. **L0 静态前缀冻结**：所有 Function Calling 工具 Schema 仅在任务初始化时注册并冻结进 L0，运行期严禁动态增删工具或修改 L0 定义。
2. **与 L3 卸载阈值解耦**：`CacheFirstContextBuilder` 的入口卸载阈值为 `2,500` 字符；工具层必须控制 `search_api` 与 `read_observation` 的正常输出严格低于该阈值，避免触发“调用工具 → 被 L3 卸载 → 再回查”的内耗闭环。
3. **不绕过终局核验**：提交格式优化与 Finish Protocol 提示不得绕过 `reflection_v1` 的拦截与业务状态校验。

---

## 2. 实证驱动：Train-15 基线诊断依据

在 `cache_first (Phase 4) + reflection_v1` 的 Train-15 基线运行（523 次工具调用，TGC 8/15，断言通过率 59/78 = 75.6%）中，离线诊断暴露出三类集中的工具层摩擦：


| 摩擦类型                          | Train-15 基线数据                                                                                                                                       | 典型影响                                                                                           |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------- |
| **探索与回查占比过高**            | `search_api` (140) + `get_api_schema` (58) + `read_observation` (73) + `search_history` (26) = **297/523 (56.8%)**                                      | 超过一半回合消耗在找接口和翻历史，多于实际干活的`execute_python` (226)。                           |
| **工具输出反噬 L3 卸载线**        | `read_observation` 平均 **2,917 字符**（52/73 次超 2,500）；`search_api` 有 **50/140 次**超 2,500                                                       | 回查旧 Observation 时反手再次触发 L3 卸载，放大多轮翻查（如`b0a8eae_2` 单题 48 次历史检索/读取）。 |
| **`execute_python` 报错高度集中** | 42/226 次报错（**18.6%**）：HTTP 422 (9)、Finish Protocol 混写 (8)、HTTP 401 (5)、位置参数错 (4)、打印 API 函数不可序列化 (4)、沙盒禁用 `csv.writer` 等 | 缺少机器可读的错误分类与部分成功状态，易引发无效重试或中途卡死（如`6104387_1`、`2a163ab_3`）。     |

---

## 3. 核心机制（TE-MVP / Phase 1）

### 3.1 Tool Discovery：紧凑签名与分页显性化

- **`search_api` 瘦身**：移除 `schema-aware` 候选里与外层重复的 `api_name` / `description` 字段，以紧凑参数格式清晰标注必填（`*`）与可选（`?`），将默认 `top_k=5` 的序列化输出稳定压在 **1,800 字符以内**。
- **分页能力标识**：借助本地 `ApiRetriever` 官方文档索引，自动检测参数含 `page_index` 或 `page_limit` 的列表接口，在 `search_api` 与 `get_api_schema` 输出中附带分页提醒（`paginated` / 遍历至空页提示），减少仅读第 0 页导致的目标集合遗漏。

### 3.2 Tool Feedback：Requester 部分成功反馈 + 错误自愈信封

- **部分成功摘要（`execution_summary`）**：复用 `AppWorldTool.execute_python()` 执行前后的 Requester 增量切片，当代码块执行失败或新增事件包含非 2xx 响应时，返回紧凑摘要：
  - `total_calls` / `succeeded_calls` / `failed_calls`
  - `succeeded_mutations`：本次异常前已成功落库的写操作精简列表（明确提示修复报错后切勿重复写入）
  - `first_failure`：首个失败 API 的 `endpoint`、`status_code` 与核心错误信息
- **Actionable Error Envelopes（`error_envelope`）**：在保留原始真实错误的前提下，附加 `error_type` 与 1–2 句可执行 `recovery_hint`，优先覆盖 5 类高频硬错：
  1. `http_422_validation`：定位失败接口并直接从本地 `ApiRetriever` 附带必填参数列表，省去额外一轮 `get_api_schema`。
  2. `http_401_auth`：提示先调用 `apis.supervisor.show_account_passwords()` 核对凭据再重新登录，阻止盲猜重试。
  3. `positional_arg_error`：提示 AppWorld API 全量强制使用关键字参数。
  4. `sandbox_restricted`：提示受限 stdlib / 内置（如 `csv.writer`、`open`）不可用，改用纯 Python 字符串构造 + `apis.file_system` 读写。
  5. `json_not_serializable`：提示避免直接打印或序列化 `apis.*` 模块与函数对象。

### 3.3 Tool Projection：防二次卸载与边界对齐

- **程序可见 vs 模型可见分离**：完整 API 返回值留在持久化 Python Shell 变量与 `ObservationStore` 中供代码直接处理；进入模型上下文的仅是受控长度的结果或回查切片。
- **消除 `read_observation` 自卸载**：将 `ObservationStore.read_observation` 默认 `limit` 收紧至 `1,600` 字符，并在 `CacheFirstContextBuilder._freeze_output()` 中为 `read_observation` 增加免二次 L3 卸载豁免。

### 3.4 提交协议文案收口

- 不新增额外 Function Calling 工具，仅在 `execute_python` 工具描述与 Finish Protocol 拦截提示中统一明确：所有写操作与状态读回确认完成后，单独提交一条 `apis.supervisor.complete_task(answer=...)`；数值问答传纯值不带多余单位，纯动作任务传 `answer=None`。

---

## 4. 代码落点


| 模块文件                                      | 职责                                                                                                                                                               |
| :-------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/tool_engineering/tools.py`      | 工具 Schema 注册、`ApiRetriever` 本地文档索引、`search_api`/`get_api_schema` 紧凑渲染与分页标记、`ToolRegistry.dispatch()` 错误自愈信封与 `execution_summary` 封装 |
| `src/environments/appworld.py`       | `AppWorldTool.execute_python()` 执行前后 Requester 增量切片与部分成功摘要采集                                                                                      |
| `src/memory/store.py`                | `ObservationStore.read_observation()` 默认页大小（`limit=1600`）与原始数据外置读取                                                                                 |
| `src/context_engineering/context.py` | `_freeze_output()` 中 `read_observation` 的免二次 L3 卸载豁免                                                                                                      |
| `src/agent/react.py`                 | Finish Protocol 拦截提示文案对齐（不改`reflection_v1` 校验逻辑）                                                                                                   |

---

## 5. 演进路线（按数据驱动开启）


| 阶段                  | 状态     | 内容                                                                                                                                                                                                                                            | 开启条件                                                          |
| :-------------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------- |
| **TE-Phase 1 (MVP)**  | 已落地   | 消除`read_observation` 二次卸载、`search_api` 紧凑签名与分页标记、Requester 部分成功反馈、5 类错误自愈信封、提交文案对齐                                                                                                                        | 已完成离线测试及 7 题定点在线复测                                  |
| **TE-Phase 2**        | 已落地   | L0 静态 App 能力目录、轻量同义词扩展、会话级 Schema 去重；不增删工具、不改变 Assurance 判定                                                                                                                                           | 当前离线单测通过；尚未进行 Phase 2 专项在线消融                    |
| **TE-Phase 3**        | 待推进   | L3 JSON 骨架预览、`read_observation` 的 `grep` / `keys_only`、按需独立 `complete_task`                                                                                                                                               | 根据 Phase 2 在线轨迹中仍存在的发现/投影/提交摩擦决定              |

---

## 6. 验证与对照指标

### 离线单测

- `tests/test_tool_engineering_mvp.py`：覆盖 `read_observation` 默认 `limit=1600` 与免二次卸载、`search_api` 长度上限与分页标记、Requester 部分成功摘要、5 类 `error_envelope`（含 422 真实必填参数反查）。
- 全量回归：`tests/test_cache_first_context.py` 及原有工具 / `execution_assurance` / `react` 单测 100% 通过。

### 在线消融对照（Train-15，同配置 `cache_first + reflection_v1`）

对比 TE-MVP 落地前后的以下硬指标，决定是否进入 TE-Phase 2：

- **准确率**：官方 TGC（基线 8/15 = 53.3%）、细粒度断言通过率（基线 59/78 = 75.6%）。
- **工具摩擦**：`execute_python` 报错率（基线 18.6%）、探索与回查调用占比（基线 56.8%）。
- **成本效率**：平均推理步数、总输入 Token、步数上限截断题数（基线 1 题）。

### TE-MVP 七题定点复测

在 TE-MVP 后，对 Train-15 基线中未全过的 7 道任务各运行一次，结果目录分为首次批次和外部启动中断后的补跑批次：

- 基线：`D:\worldapp\runs\agentlab_phase4_cache_first_reflection_train15_20261125_225037`
- TE-MVP：`D:\worldapp\runs\agentlab_te_mvp_failed7_20261125_235108` 与 `D:\worldapp\runs\agentlab_te_mvp_failed7_external_retry_20261125_235351`

| 指标 | 基线 | TE-MVP 七题复测 |
| :--- | ---: | ---: |
| 完整通过任务 | 0/7 | 4/7 |
| 断言通过 | 26/45 (57.8%) | 33/45 (73.3%) |
| 平均步数 | 47.0 | 28.9 |
| 达到 100 步上限 | 1 | 0 |
| 主模型 + Clone 输入 Token | 2,974,179 | 1,971,371 (-33.7%) |
| `read_observation` 调用 | 57 | 8 |

各题断言变化：`6ea6792_2` 5/6→5/6、`e7a10f8_1` 1/2→2/2、`2a163ab_3` 2/6→6/6、`60d0b5b_3` 6/7→7/7、`6104387_1` 1/10→1/10、`afc0fce_2` 7/9→7/9、`b0a8eae_2` 4/5→5/5。

这是每题单次运行的小样本定点复测，不构成统计显著性或单独证明 TE-MVP 导致全部收益；模型随机性、运行批次与工具改动的影响不能从这组数据完全拆分。Phase 2 的三项发现侧优化尚未在线单独验证。

## 7. TE-Phase 2 实现说明

### L0 静态 App 能力目录

`LangGraphAgent.solve()` 在任务初始化时根据该任务 `allowed_apps` 构建一次 `ApiRetriever`。`app_capability_catalog()` 仅读取此检索器已过滤的官方 API 文档，按 API 名称和官方描述确定性提取高频能力关键词，生成每个应用一行的短目录，并通过 `ContextManager` 写入 System Prompt。该目录随初始 L0 冻结；运行期间不重算、不改写。相同 `ApiRetriever` 实例传给 `AppWorldTool`，避免目录和实际搜索索引不一致。能力目录只作导航提示，API 参数仍以搜索候选 Schema 或 `get_api_schema` 官方文档为准。

### 查询别名扩展

`ApiRetriever.search_api()` 在 BM25 计算前追加少量通用同义词组，覆盖删除/关闭、支付/转账、备注、歌曲/播放列表、备份/导出/文件、联系人/同事等概念。它只扩展查询词，不更改文档索引、候选排序算法、BM25 原始分数语义、Schema 渲染、1,800 字符上限或分页标记。

### Schema 会话去重

每个任务的 `ToolRegistry` 独立维护 `seen_schemas`。首次成功查询保留原官方 Schema；同会话重复查询同一规范 `app.api` 时，返回必填参数签名、分页标记（适用时）和 `already_fetched_in_session: true`，不重复发送长描述。新任务会创建新 Registry，因此不会跨任务共享缓存或状态。Function Calling 工具定义与工具数量保持不变。

## 8. Phase 2 验证范围

- 新增 `tests/tool_engineering/test_tool_engineering_phase2.py`：验证任务运行中的 L0 目录逐步字节不变且工具列表不变、同义词检索和紧凑分页结果、Schema 重复查询压缩及任务级 Registry 隔离。
- 当前定点测试结果：3 项通过。Phase 2 在线增益尚未测试；不将 TE-MVP 七题复测归因到本阶段代码。
- 回归测试需覆盖 Tool Engineering、Agent、Context Engineering、Execution Assurance、AppWorld 环境及 Cache-First Phase 1–4，见本次交付测试报告。
