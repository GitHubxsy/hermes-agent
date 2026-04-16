# 从零开始理解 Agent 进阶篇（一）：hermes 的循环骨架

> 前置知识：nanoAgent 系列第一篇——你已理解 `Agent = LLM + 工具 + 循环`，知道 `agent.py` 的 115 行做了什么。本文在此基础上，打开 hermes 的同一层代码，看它如何把这 115 行放大成工业级实现。

---

## 一、先回忆 nano 的骨架

```python
# agent.py — 核心循环，约 30 行
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
messages.append({"role": "user", "content": user_input})

while True:
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS
    )
    msg = response.choices[0].message

    if not msg.tool_calls:          # 没有工具调用 → 任务完成
        print(msg.content)
        break

    messages.append(msg)            # 保留 assistant 消息

    for tc in msg.tool_calls:       # 逐个执行工具
        result = call_tool(tc.function.name, tc.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
```

三件事：**问模型 → 执行工具 → 把结果塞回消息列表**，循环直到没有工具调用为止。

hermes 做的是同一件事，但要应对真实世界的复杂性：多线程、异常恢复、资源超支、并发工具、多 provider……下面逐层拆开。

---

## 二、找到 hermes 的同一段代码

hermes 的循环在 `run_agent.py` 的 `AIAgent` 类里，入口是 `run_conversation()`。

nano 的 `while True` 在 hermes 里变成：

```python
# run_agent.py（简化展示）
while api_call_count < self.max_iterations \
      and self.iteration_budget.remaining > 0:

    # ① 调用 API
    response = client.chat.completions.create(...)

    # ② 有工具调用 → 执行，循环
    if assistant_message.tool_calls:
        messages.append(assistant_msg)
        tool_results = self._execute_tool_calls(assistant_message.tool_calls)
        messages.extend(tool_results)
        api_call_count += 1
        continue

    # ③ 无工具调用 → 返回最终答案
    return {"final_response": assistant_message.content, "messages": messages}
```

结构完全一致。不同的是循环条件从 `True` 变成了两个约束，以及中间多出大量"护栏"。

---

## 三、第一层扩展：循环预算（IterationBudget）

nano 的 `while True` 永远不会自己停下来——如果工具一直返回结果，它会无限循环下去，把 API 额度烧光。

hermes 的解法是 `IterationBudget`（`run_agent.py:170`）：

```python
class IterationBudget:
    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()   # ← 线程安全

    def consume(self) -> bool:
        with self._lock:
            if self._used >= self.max_total:
                return False            # ← 超限，拒绝继续
            self._used += 1
            return True

    def refund(self) -> None:           # ← 某些工具可以退回预算
        with self._lock:
            if self._used > 0:
                self._used -= 1
```

**为什么要有 `refund()`？**

`execute_code` 工具允许模型自己写代码调用工具。这些"程序化工具调用"不消耗人类意图，只是执行层的实现细节，不应算在"agent 思考次数"里。调用完后 `refund()` 归还一次预算。

**为什么要加锁？**

hermes 支持子 agent（delegate_tool），子 agent 运行在独立线程里。每个 agent 有自己的 `IterationBudget` 实例，但**父 agent 可以把自己的 budget 实例传给子 agent 共享**——这时候多个线程同时 `consume()`，没有锁会有竞争条件。

```python
# AIAgent.__init__:
self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
#                       ↑ 子 agent 传入父级 budget，共享计数
```

---

## 四、第二层扩展：工具注册表

nano 的工具是硬编码的函数字典：

```python
# agent.py
TOOLS = [execute_bash_schema, read_file_schema, write_file_schema]

def call_tool(name, args):
    if name == "execute_bash": return execute_bash(args)
    if name == "read_file":    return read_file(args)
    if name == "write_file":   return write_file(args)
```

hermes 有几十个工具，不可能都堆在一个文件里。解法是**中央注册表** `tools/registry.py`：

```python
class ToolEntry:
    __slots__ = (
        "name", "toolset", "schema", "handler",
        "check_fn",          # ← 运行时可用性检查
        "requires_env",      # ← 需要哪些环境变量
        "is_async",
        "max_result_size_chars",  # ← 工具输出大小上限
    )

class ToolRegistry:
    def register(self, name, toolset, schema, handler,
                 check_fn=None, requires_env=None, ...):
        self._tools[name] = ToolEntry(...)
```

每个工具文件在**模块加载时**自注册：

```python
# tools/web_tools.py（举例）
registry.register(
    name="web_search",
    toolset="web",
    schema={...},
    handler=lambda args, **kw: web_search(args["query"]),
    check_fn=lambda: bool(os.getenv("SEARCH_API_KEY")),  # ← 没 key 就不出现
    requires_env=["SEARCH_API_KEY"],
)
```

`check_fn` 是关键设计：**工具只有在运行环境满足条件时才会出现在 LLM 的工具列表里**。没有配置 API key 的工具不会出现，模型不会幻觉调用它。

---

## 五、第三层扩展：工具并行执行

nano 的工具是顺序执行的——一个跑完再跑下一个。

```python
for tc in msg.tool_calls:   # 串行
    result = call_tool(...)
```

模型有时会在一次响应里同时调用多个**独立**工具，比如"同时读三个文件"、"同时搜索两个关键词"。串行执行浪费时间。

hermes 用 `_should_parallelize_tool_batch()`（`run_agent.py:267`）判断是否可以并行：

```python
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})  # 需要用户输入，不能并发

_PARALLEL_SAFE_TOOLS = frozenset({
    "read_file", "search_files",
    "web_search", "web_extract",
    "session_search", "vision_analyze",
    ...  # 只读、无共享状态
})

_PATH_SCOPED_TOOLS = frozenset({
    "read_file", "write_file", "patch"
})  # 文件工具：目标路径不重叠时可以并行
```

判断逻辑：

```python
def _should_parallelize_tool_batch(tool_calls) -> bool:
    if len(tool_calls) <= 1:
        return False
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False          # 有交互工具 → 串行

    for tool_call in tool_calls:
        if tool_name in _PATH_SCOPED_TOOLS:
            path = _extract_parallel_scope_path(tool_name, args)
            if any(_paths_overlap(path, existing) for existing in reserved_paths):
                return False  # 操作同一文件 → 串行
            reserved_paths.append(path)
            continue
        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False      # 未知安全性 → 保守串行
    return True
```

通过检查后用 `ThreadPoolExecutor` 并发执行，最后收集结果按原顺序返回给模型。

---

## 六、第四层扩展：错误恢复

nano 对 JSON 解析错误直接崩溃。hermes 有两种恢复机制：

**恢复一：无效 JSON 重试**（`run_agent.py:9880`）

```python
if invalid_json_args:
    if self._invalid_json_retries < 3:
        continue   # 直接重试 API call，不追加任何消息

    else:
        # 超过 3 次：把错误作为 tool result 注入，让模型自我修复
        for tc in assistant_message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Error: Invalid JSON. Please retry with valid JSON.",
            })
        continue
```

**恢复二：不完整推理块重试**

某些模型支持 `<REASONING_SCRATCHPAD>` 推理标签。如果模型输出在推理块中途被截断（输出 token 耗尽），hermes 检测到后最多重试 2 次：

```python
if has_incomplete_scratchpad(assistant_message.content or ""):
    self._incomplete_scratchpad_retries += 1
    if self._incomplete_scratchpad_retries <= 2:
        continue  # 重试，不把截断的消息加入历史
```

---

## 七、循环如何结束

nano 只有一种结束：没有工具调用。hermes 有五种：

| 终止条件 | 触发位置 |
|---------|---------|
| 没有工具调用（正常完成） | 主循环 else 分支 |
| `api_call_count >= max_iterations` | `while` 条件 |
| `iteration_budget.remaining == 0` | `while` 条件 |
| 用户中断（`Ctrl+C` 或消息平台 `/stop`） | `tools/interrupt.py` 设置中断标志 |
| 上下文窗口超限触发压缩后继续 | `ContextCompressor`（第六篇对应模块）|

---

## 八、整体流程图

```
用户消息
    ↓
┌─────────────────────────────────────────────────┐
│  while budget.remaining > 0 and count < max     │
│                                                  │
│  ① 组装 messages + tools（注册表过滤）            │
│       ↓                                          │
│  ② API call（支持 OpenAI / Anthropic / 本地模型）│
│       ↓                                          │
│  ③ 响应格式归一化（OpenAI / Anthropic / Codex）  │
│       ↓                                          │
│  ④ 有 tool_calls？                               │
│     ├── 是 → 并行安全检查                        │
│     │         ├── 安全 → ThreadPoolExecutor 并发 │
│     │         └── 不安全 → 串行执行              │
│     │        错误恢复（JSON / 截断推理）          │
│     │        追加 tool results → 继续循环        │
│     └── 否 → 返回最终答案                        │
└─────────────────────────────────────────────────┘
```

---

## 小结：nano → hermes 的演进

| 概念 | nano（115行） | hermes |
|------|-------------|--------|
| 循环终止 | `while True`，永不停 | `IterationBudget`，有上限、可退款、线程安全 |
| 工具定义 | 硬编码字典 | `ToolRegistry`，模块加载时自注册，`check_fn` 按需启用 |
| 工具执行 | 串行 | 安全检查后并行（`ThreadPoolExecutor`） |
| 错误处理 | 崩溃 | JSON 重试 3 次 → 注入错误 tool result → 模型自修复 |
| 多 provider | 无 | 响应归一化层（OpenAI / Anthropic / Codex 统一格式） |

**下一篇**将解剖**记忆系统**：`MemoryManager`、`BuiltinMemoryProvider`，以及 hermes 为什么把记忆注入为 user 消息而不是 system prompt。

---

**关键文件速查**

| 文件 | 对应内容 |
|------|---------|
| `run_agent.py:170` | `IterationBudget` 类 |
| `run_agent.py:214` | `_PARALLEL_SAFE_TOOLS` 并行白名单 |
| `run_agent.py:267` | `_should_parallelize_tool_batch()` |
| `run_agent.py:526` | `AIAgent` 类定义 |
| `run_agent.py:9880` | 无效 JSON 恢复逻辑 |
| `tools/registry.py:24` | `ToolEntry` + `ToolRegistry` |
| `model_tools.py:234` | `get_tool_definitions()` toolset 过滤 |
