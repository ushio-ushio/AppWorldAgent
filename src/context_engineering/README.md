# Cache-First Context Engineering Architecture

## 1. 设计目标与核心原则

本模块实现一套 **Cache-First Context Engineering（缓存优先的上下文工程）**，在不改动 Planner、ReAct 主循环和 Execution Assurance/Reflection 逻辑的前提下，解决长程任务中 KV Cache 命中率极低（1%–3%）与历史报错/原始大输出污染注意力的问题。

四条核心设计原则：
1. **Cache First**：固定前缀、入口冻结表示、正常执行期严格只追加，最大化 SIWC `/v1/responses` 请求序列的前缀字节复用。
2. **Information First**：分层分片、大输出入口卸载到 L3 延迟加载，保留可核验事实与写操作记录，而非原始执行流水账。
3. **Asynchronous Extraction**：确定性提取写操作/实体账本，结合高水位预触发、前缀共享的后台 Clone Agent 异步提取语义记忆，解耦提取与裁剪。
4. **Safe Compaction**：以不可变分片（Immutable Shard）批量提交主记忆，按原子 `call_id` 配对与双游标安全裁剪 L2，**有真实副作用的写操作及修复记录永不擦除**。

---

## 2. 分层内存模型与 SIWC Payload 布局

针对 `siwc_responses.py` 将工具定义、System Prompt、任务和历史按顺序序列化进单一 `input` 数组的特性，`cache_first` 模式严格按以下顺序组装每轮 `model_messages`：

```text
+-----------------------------------------------------------------------+
| L0 · Static Prefix（全程字节级永久冻结）                                 |
|   [0] additional_tools (developer, 固定排序 Tool Schema)               |
|   [1] System Prompt (developer, 含防跳步静态声明，零动态字段)            |
|   [2] Original Task + Constraints + Initial Read-only Plan (user)     |
+-----------------------------------------------------------------------+
| L1 · Main Memory（不可变只追加分片，WAL 风格）                           |
|   [Shard #1: Steps 1–k1]   <- 提交后永久冻结，永不原地改写               |
|   [Shard #2: Steps k1+1–k2]                                           |
+-----------------------------------------------------------------------+
| L2 · Working Context（严格只追加工作区 + 受控裁剪）                      |
|   - 入口已冻结的 Assistant(function_call) + Tool(output) 成对历史       |
|   - 无真实副作用的已解决报错：解决后按成对 call_id 即时物理裁剪           |
|   - 高水位 Compact 时：从头部整段切掉已归档至 L1 的连续区间              |
+-----------------------------------------------------------------------+
| Volatile Tail（易变尾部，永远置于最后一条，不污染前方任何前缀）           |
|   - 最新未解决清单 (unresolved)                                         |
|   - 本轮动态提示（Assurance 状态摘要 / 当前步数或预算提醒）              |
+-----------------------------------------------------------------------+

[ L3 · External Storage（上下文之外，按需延迟加载）]
  Phase 1 复用 `memory/store.py` 的任务级 `ObservationStore` 接口（保留
  Observation ID、任务隔离与 `read_observation` / `search_history`）。当前
  Store 本身只驻留进程内存，并非耐久外部数据库；Trace 是独立审计文件，
  不作为 Store 的自动恢复后端。超长输出入口卸载并向模型呈现永久冻结的句柄。
```

Phase 1 实际请求布局是 `L0 + L2 + Volatile Tail`；`L1` 的不可变分片尚未实现，
仅在 Phase 2 接入。文档图中的 L1 位置用于说明后续归档边界。

> 注：Reflection 只读探查期间 `react.py` 会临时收窄工具列表为 `execute_python`（影响 `[0]`）；该行为属于低频终局路径，保留原逻辑不动以维持 Reflection 零侵入。

## 3. 信息处理与即时安全擦除规则


| **信息类型**                   | **L0 / L1（长期）** | **L2（工作区）**       | **L3（外部存储）** | **处理规则**                           |
| ------------------------------ | ------------------- | ---------------------- | ------------------ | -------------------------------------- |
| 原始任务、约束与初始 Plan      | L0 永久冻结         | —                     | 保留               | 永不修改或压缩                         |
| 实际写操作及结果 (`mutations`) | L1 确定性保留       | 保留至归档             | 保留               | **含副作用的错误及修复绝不擦除**       |
| 有效业务事实、实体 ID、进展    | L1 不可变分片       | 保留至归档             | 保留               | 必须关联有效`evidence_ref`             |
| 大规模原始 Observation         | L1 仅保留句柄       | 入口即冻结为摘要句柄   | 完整保存           | 写入 L2 时一次定型，后续不回改         |
| **无副作用的已解决报错**       | 不保留              | **解决后即时原子裁剪** | 保留               | 必须同时满足零写副作用与成对`call_id`  |
| 无效只读探索 / 已纠正推理      | 不保留              | Compact 时整段清扫     | 保留               | 正常执行期不打断前缀，Compact 一并清理 |
| 未解决的问题 (`unresolved`)    | 不写死进 L1         | Volatile Tail 覆盖更新 | 保留               | 放尾部实时刷新，避免旧 L1 分片产生误导 |

### 无副作用报错即时裁剪铁律

1. **Zero-Mutation Guard**：仅当某错误步骤经 `RequesterEventParser` 确认**未产生任何成功的写操作（POST/PUT/PATCH/DELETE 等）**，且后续步骤成功完成修复时，才允许将其标记为可裁剪。
2. **Atomic Pair Deletion**：裁剪必须以完整的 `(assistant tool_calls + 对应全部 tool outputs)` 为最小原子单位移除，严格保持 `tool_call_id` 配对完整，防止 Responses API 400 报错。
3. **防跳步疑惑机制**：
   * 保留原始 `Step k` 编号，绝不重排历史消息编号；
   * 在 `L0` 静态 System Prompt 中写入固定说明：告知模型已解决的无副作用瞬态报错和已归档旧步骤会被自动清理，非连续步号属正常现象，应以 L1 分片和当前保留消息为准，切勿重复执行已归档步骤；
   * 高水位 Compact 裁剪前段区间时，在 `L2` 头部保留一行静态锚点（如 `[Context Boundary: Steps 1-11 archived into L1 Shard #1]`）。

## 4. Phase 2 确定性计量与 Epoch Compaction

Phase 2 在 cache_first 运行路径加入轻量 Token Meter 和不可变 L1 Checkpoint/Shards；旧 `TrajectoryCompactor` 与其他 context mode 不变。

- **Token Meter**：记录当前请求估算 Token、上次真实 `input_tokens` 校准值、相邻序列化请求 LCP 字符数/比例、服务端实际 `cached_tokens`、剩余容量、Compact 级别及 L1 shard 数。字符估算不是 Token 精确值。默认软触发 14,000、容量参考 20,000，可配置。
- **L1**：按顺序追加不可变 `CheckpointShard`，明确标记 `event_range`，冻结写入/修复证据及 Observation 索引；分片只记录执行证据，不推断任务目标已完成。
- **Recent Tail Protection**：默认保护最近 4 个完整工具交互组。
- **三级阶梯**：Level 1 确定性清扫无副作用且有证据支持的错误/冗余交互；重新计量后仍超水位才进入 Level 2，将保护区外连续安全头部提交为 L1 shard 并从 L2 原子移除。Level 3 属 Phase 4，尚未实现。
- **Phase 4 路线**：Level 3 整体上下文重建、Overflow Recovery 与轻量 Request Replay；Phase 3 在软水位后台异步提取语义记忆，使用同一 `session-id` 和主 Agent 的稳定 L0/L1/L2 前缀；证据引用校验后由双游标限制归档边界。Clone 不执行工具、不授予 verified，失败或未就绪时降级为 Phase 2 确定性归档。终局提交或 Reflection 会忽略在途结果，不等待后台请求。Phase 4 尚未实现。

平台约束：SIWC 使用 `store=false` 且不传 `prompt_cache_key` 时，服务端缓存粘性不能保证。固定 `session-id` 作为同任务请求关联提示进行实测，但不视为缓存命中保证；确定性降噪和裁剪仍可独立降低输入上下文。

## 5. 双轨结构化提取与解耦裁剪（Phase 3）

### 4.1 结构化提取 Schema

**JSON**

```
{
  "covered_step_range": [1, 12],
  "facts": [{ "text": "...", "evidence_ref": "obs_xxx / step_k" }],
  "entities": [{ "type": "...", "id": "...", "note": "...", "evidence_ref": "..." }],
  "progress": [{ "text": "...", "evidence_ref": "..." }],
  "mutations": [{ "api": "...", "status": "ok/err", "summary": "...", "evidence_ref": "..." }],
  "unresolved": [{ "text": "...", "evidence_ref": "..." }]
}
```

* **确定性轨道（0 Token）**：`mutations` 与基础写操作实体直接由 Requester 事件确定性生成，自动标为 `verified`。
* **异步语义轨道（Clone Agent）**：不每步触发，仅在 L2 达到软水位（Soft Watermark）时后台异步触发；Clone 请求字节级复用主 Agent 当前 `L0 + L1 + L2` 前缀（Prefix Forking）仅在末尾附提取指令以复用缓存。Clone 不执行工具，也不能将条目自标为 `verified`。

### 4.2 双游标安全 Commit 与 Compact

* 维护 `extracted_cursor`（Pending Memory 已覆盖的最大步数）与尾部保留窗口 `keep_recent_steps`（默认 4 步，按完整 `call_id` 组对齐）。
* 仅当 L2 触及硬水位（Hard Watermark）时触发同步 Commit + Compact，允许归档并从 L2 头部切掉的最大右边界为： `compact_end = min(extracted_cursor, current_step - keep_recent_steps)`
* Commit 阶段校验每条候选记忆是否持有合法 `evidence_ref`（存在于对应区间的 step/obs ID 中），无引用条目降级或丢弃；随后冻结追加为新的不可变 `L1 Shard`，再将 `L2` 对应头部连续区间整体移除。
* 主 Agent 进入 `complete_task` 时，自动取消/忽略尚未返回的后台 Clone 任务，零阻塞终局提交与 Reflection。

## 5. 代码结构与兼容边界

为保持向后兼容与严格 A/B 对照，保留既有 `dynamic.py`、`unified.py` 及 `compaction.py` 中的 `TrajectoryCompactor` 不变；新增独立模式 `context_mode="cache_first"`：

**Plaintext**

```
src/context_engineering/
├── README.md        # 本架构规格书
├── context.py       # [Phase 1] CacheFirstContextBuilder：L0冻结、L2只追加、L3入口卸载、即时报错裁剪、Volatile Tail
├── compaction.py    # [Phase 2] 保留旧 TrajectoryCompactor；新增不可变 L1 Shard 提交与双游标 EpochCompactor
├── extraction.py    # [Phase 3] 确定性账本汇总 + 软水位预触发、前缀共享后台 Clone Agent 与 PendingMemory
├── dynamic.py       # [Legacy 对照保留]
└── unified.py       # [Legacy 对照保留]
```

**不可破坏的外部接口边界**：

* `state["messages"]` 仍完整记录原始历史，供审计与调试；`cache_first` 仅负责生成缓存稳定的 `model_messages` 视图；
* 严格保留 `tool_call_id / call_id` 成对顺序、Observation ID、`[Pagination Hint]`、`RequesterEventParser` 事件、Reflection 证据包及 `AGENTLAB_BATCH_CHECK` 逻辑不变。

## 6. 分阶段演进与评测指标

* **Phase 1（`context.py`）**：落地 `L0` 冻结 + `L2` 严格只追加 + `L3` 入口卸载冻结 + 无副作用报错即时裁剪 + `Volatile Tail`（零额外 LLM 调用）。目标：验证 KV Cache 命中率从 1%–3% 显著跃升，准确率不降。
* **Phase 2（`compaction.py` / `context.py`）**：确定性 Token Meter、不可变 L1 Shard、Recent Tail Protection 和两级裁剪。
* **Phase 3（`extraction.py`）**：异步前缀共享 Clone Agent 补充语义记忆。
* **Phase 4**：Level 3 整体重建、Overflow Recovery 与轻量 Request Replay。

核心追踪指标：`cached_tokens / input_tokens`（首要）、未缓存输入 Token、总 Token（含 Clone）、端到端延迟、任务通过率与跨次方差。


## Phase 4 高压恢复与离线诊断

Phase 4 在 cache_first 独立路径增加高压 Level 3：当上下文估算达到硬容量的可配置比例时，将 L1 合并为单个 `CP-MASTER-01`，保留写操作证据、实体、事实、未解决项和 Observation ID 索引。SIWC 明确报告上下文超限时，Runtime 在同一 ReAct 步内执行有界紧急压缩并重放请求；传输层瞬态错误继续由 SIWC 传输重试负责。

后台 Clone 提取受增量节流：越过软水位后，默认至少新增 4 组交互且估算输入增长 3,500 Token 才再次提交。使用 `python -m context_engineering.replay <trace.jsonl>` 做离线相邻步 LCP、压缩次数、溢出恢复及缓存效率诊断。Trace 未含原始请求快照时，工具只报告可从现有 Meter 可靠恢复的指标，不伪造首差异位置。
