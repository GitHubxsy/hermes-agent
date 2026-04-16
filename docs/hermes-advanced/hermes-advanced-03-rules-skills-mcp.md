# 从零开始理解 Agent 进阶篇（三）：hermes 的 Rules、Skills 与 MCP

> 前置知识：nanoAgent 系列第三篇——你已理解 `agent-claudecode.py` 如何通过 `.agent/rules/*.md` 注入行为约束、`.agent/skills/*.json` 提供领域知识、MCP 协议扩展工具能力。本文对照 hermes 的实现，看这三层在工业场景下如何被分别扩展。

---

## 一、先回忆 nano 的方案

```python
# agent-claudecode.py — Rules（约 10 行）
def load_rules() -> str:
    rules_dir = Path(".agent/rules")
    if not rules_dir.exists():
        return ""
    return "\n\n".join(
        f.read_text() for f in sorted(rules_dir.glob("*.md"))
    )

# Skills（约 15 行）
def load_skill(name: str) -> str:
    skill_file = Path(f".agent/skills/{name}.json")
    if not skill_file.exists():
        return ""
    return json.loads(skill_file.read_text())["content"]

# MCP（约 30 行）
def call_mcp_tool(server_cmd, tool_name, args):
    proc = subprocess.run(server_cmd, input=json.dumps({...}), capture_output=True)
    return json.loads(proc.stdout)
```

nano 的方案解决了"能用"的问题，但留下了三个工程问题：

1. **Rules 是盲注入**：任何 `.md` 文件内容都直接进系统提示，包括恶意内容
2. **Skills 索引靠模型记忆**：LLM 必须自己记住有哪些技能可用，而不是在提示中看到
3. **MCP 是一次性调用**：没有持久连接，没有重连，每次工具调用都重启子进程

---

## 二、Rules → 上下文文件扫描

hermes 没有"rules 目录"这个概念，而是**识别现有生态里的标准文件**：

```python
# agent/prompt_builder.py:998
def build_context_files_prompt(cwd=None, skip_soul=False) -> str:
    """
    优先级（第一个命中即停止，只加载一种项目上下文）：
      1. .hermes.md / HERMES.md  — hermes 专属，向 git root 向上搜索
      2. AGENTS.md / agents.md   — 仅当前目录
      3. CLAUDE.md / claude.md   — 仅当前目录
      4. .cursorrules / .cursor/rules/*.mdc — 仅当前目录
    
    SOUL.md（全局身份文件）独立存在，始终加载，不受上面优先级影响。
    """
```

**与 nano 的关键区别一：注入前扫描**

每个文件加载后，先通过 `_scan_context_content()` 过滤（`agent/prompt_builder.py:55`）：

```python
_CONTEXT_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user',                       "deception_hide"),
    (r'system\s+prompt\s+override',                          "sys_prompt_override"),
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET)',             "exfil_curl"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc)',             "read_secrets"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)',   "html_comment_injection"),
    (r'<\s*div\s+style\s*=.*display\s*:\s*none',            "hidden_div"),
    ...  # 共 10 种模式
]

# 还检测不可见 Unicode（零宽字符、方向控制符等）
_CONTEXT_INVISIBLE_CHARS = {'\u200b', '\u200c', '\u200d', '\u202e', ...}
```

被拦截的文件不会加载，替换为说明文字告知模型文件被屏蔽及原因——不会静默失败。

**与 nano 的关键区别二：全局 SOUL.md**

除了项目级上下文文件，hermes 支持一个**全局身份文件** `~/.hermes/SOUL.md`，用来定义 agent 的个性、口吻、价值观：

```
~/.hermes/SOUL.md        # 全局：所有项目共享的 agent 特质
./AGENTS.md              # 项目级：这个仓库的规则和约定
```

SOUL.md 会被优先加载到"身份槽"（identity slot），然后 `build_context_files_prompt(skip_soul=True)` 跳过它，避免重复注入。

**每个文件限制 20,000 字符**，超出部分保留前 70% + 后 20%，中间替换为截断提示。

---

## 三、Skills → 两层缓存的索引系统

nano 把技能文件完整内容注入提示。hermes 的方案更精细：**系统提示只放索引，模型按需加载完整内容**。

### 技能文件结构

```
~/.hermes/skills/
└── deploy/
    ├── SKILL.md          # 技能主文件（YAML frontmatter + Markdown 正文）
    ├── DESCRIPTION.md    # 分类描述（可选）
    └── references/       # 附件目录（模板、脚本等）
        └── deploy.sh
```

`SKILL.md` 的 frontmatter：

```yaml
---
name: deploy
description: Deploy to production via blue-green strategy
platforms: [linux, macos]
metadata:
  hermes:
    requires_tools: [terminal]          # 没有 terminal 工具就不显示
    fallback_for_toolsets: [browser]    # 有 browser 时隐藏（有更好的方案）
    config:
      - key: deploy.target_host
        description: SSH hostname for production server
---

## Steps
1. Run `./scripts/health_check.sh`
2. ...
```

### 注入到系统提示的只是索引

`build_skills_system_prompt()`（`agent/prompt_builder.py:575`）生成的内容类似：

```
## Skills (mandatory)
Before replying, scan the skills below. If a skill matches or is even 
partially relevant to your task, you MUST load it with skill_view(name)...

<available_skills>
  deploy:
    - deploy: Deploy to production via blue-green strategy
  testing:
    - pytest-workflow: Run pytest with coverage and generate reports
    - e2e-cypress: End-to-end browser testing with Cypress
</available_skills>
```

索引只有名称和一行描述。**完整技能内容需要模型调用 `skill_view(name)` 才能获取**。这样做：
- 系统提示 token 消耗恒定（不随技能数量增长）
- 加载的技能只出现在对话里，不会污染其他会话的缓存

### 两层缓存

扫描技能目录是 I/O 密集操作，尤其是技能多时。hermes 用两层缓存解决启动延迟（`agent/prompt_builder.py:599`）：

```
Layer 1：in-process LRU dict（8 个槽）
  key = (skills_dir, external_dirs, available_tools, available_toolsets, platform)
  命中 → 直接返回，零 I/O

Layer 2：disk snapshot（.skills_prompt_snapshot.json）
  存储每个 SKILL.md 的 mtime + size 指纹
  进程重启后如果文件没变 → 直接读 JSON，跳过全量扫描
  文件变了 → 触发冷路径全量扫描 + 写入新 snapshot
```

### 条件激活：技能按上下文自动显示/隐藏

frontmatter 里的 `requires_tools` / `fallback_for_toolsets` 让技能索引随可用工具动态变化：

```python
# agent/prompt_builder.py:544
def _skill_should_show(conditions, available_tools, available_toolsets) -> bool:
    # fallback_for: 主工具可用时隐藏（有更好的替代方案）
    for ts in conditions.get("fallback_for_toolsets", []):
        if ts in available_toolsets:
            return False

    # requires: 必要工具不可用时隐藏
    for t in conditions.get("requires_tools", []):
        if t not in available_tools:
            return False

    return True
```

例如：`web-fallback` 技能在 `browser` toolset 可用时自动隐藏；`deploy` 技能在没有 `terminal` 工具时不出现在索引里。

### `/技能名` 调用：注入为 user 消息

用户输入 `/deploy` 时，CLI 和 Gateway 都调用 `skill_commands.py` 里的同一段逻辑：加载技能内容，**以 user 消息的形式注入**，而不是修改 system prompt。

原因跟记忆系统一样：system prompt 改动会让 prompt cache 失效。user 消息每轮都不同，注入到这里不影响缓存。

---

## 四、MCP → 持久连接的工业级客户端

nano 的 MCP 是"用完即弃"：每次工具调用都重启子进程，建立连接，取工具，关闭。hermes 的 MCP 客户端（`tools/mcp_tool.py`，1050 行）完全不同。

### 架构：独立后台事件循环

```python
# tools/mcp_tool.py:1070
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None

def _ensure_mcp_loop():
    """启动后台事件循环线程（只启动一次）"""
    global _mcp_loop, _mcp_thread
    _mcp_loop = asyncio.new_event_loop()
    _mcp_thread = threading.Thread(
        target=_mcp_loop.run_forever,
        daemon=True,        # ← 随主进程退出，不阻塞关闭
        name="mcp-event-loop",
    )
    _mcp_thread.start()
```

所有 MCP 服务器的异步操作都跑在这个专属事件循环上。主线程（agent 循环）通过 `run_coroutine_threadsafe()` 把工具调用提交进去，拿到 `Future` 后阻塞等待结果。

### MCPServerTask：每个服务器一个长生命周期 Task

```python
# tools/mcp_tool.py:720
class MCPServerTask:
    """一个 MCP 服务器连接的完整生命周期管理"""
    
    __slots__ = (
        "name", "session",          # 连接名称和当前会话
        "_ready",                    # asyncio.Event：连接就绪信号
        "_shutdown_event",           # asyncio.Event：主动关闭信号
        "_tools",                    # 已发现的工具列表
        "_sampling",                 # Sampling handler（服务器主动发起 LLM 调用）
        ...
    )
```

**自动重连 + 指数退避**（`tools/mcp_tool.py:986`）：

```python
async def run(self, config: dict):
    retries = 0
    backoff = 1.0       # 初始等待 1 秒

    while True:
        try:
            await self._run_stdio(config)  # 或 _run_http
            break  # 正常退出（主动 shutdown）

        except Exception as exc:
            if retries > _MAX_RECONNECT_RETRIES:  # 5 次
                logger.warning("MCP server '%s' gave up after %d retries", ...)
                return

            retries += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)  # 最大 60 秒
```

### 三种传输方式

```yaml
# ~/.hermes/config.yaml 配置举例
mcp_servers:
  filesystem:
    command: "npx"          # stdio 传输
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

  remote_api:
    url: "https://my-server.example.com/mcp"  # HTTP/StreamableHTTP 传输
    headers:
      Authorization: "Bearer sk-..."
    timeout: 180

  analysis:
    command: "npx"
    args: ["-y", "analysis-server"]
    sampling:               # 服务器主动发起 LLM 请求
      enabled: true
      model: "gemini-3-flash"
      max_tokens_cap: 4096
```

### 安全：两层防护

**防护一：环境变量过滤**（`tools/mcp_tool.py:192`）

stdio 子进程只继承安全的基础环境变量，不泄露 API 密钥：

```python
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
})

def _build_safe_env(user_env: Optional[dict]) -> dict:
    env = {}
    for key, value in os.environ.items():
        if key in _SAFE_ENV_KEYS or key.startswith("XDG_"):
            env[key] = value  # 只传基础变量
    if user_env:
        env.update(user_env)  # 再叠加用户显式配置的变量
    return env
```

**防护二：错误消息凭证剥离**（`tools/mcp_tool.py:211`）

工具执行出错时，错误消息返回给 LLM 之前先过滤掉凭证信息：

```python
_CREDENTIAL_PATTERN = re.compile(
    r"(?:"
    r"ghp_[A-Za-z0-9_]{1,255}"    # GitHub PAT
    r"|sk-[A-Za-z0-9_]{1,255}"    # OpenAI-style key
    r"|Bearer\s+\S+"               # Bearer token
    r"|token=[^\s&,;\"']{1,255}"  # token=...
    r"|password=[^\s&,;\"']{1,255}"
    r")",
    re.IGNORECASE,
)

def _sanitize_error(text: str) -> str:
    return _CREDENTIAL_PATTERN.sub("[REDACTED]", text)
```

### 动态工具发现

MCP 协议支持服务器在运行时发送 `tools/list_changed` 通知。hermes 订阅了这个通知，当收到时自动重新拉取工具列表并更新注册表（`tools/mcp_tool.py:787`）：

```python
async def _refresh_tools(self):
    """服务器发出 tools/list_changed 时调用"""
    async with self._refresh_lock:           # 防止并发刷新
        new_tools = await session.list_tools()
        # 对比旧列表：注销消失的工具，注册新出现的工具
        ...
        registry.unregister(disappeared_tools)
        registry.register(new_tools)
```

nano 没有这个能力——服务器更新工具后必须重启 agent。

---

## 五、整体架构图

```
                    system prompt 组装
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    Rules（上下文文件）  Skills 索引   MCP 工具列表
          │              │              │
   优先级扫描       两层缓存加速    后台事件循环
   注入检测        条件激活过滤    持久连接
   20k 字符上限    disk snapshot   自动重连
                         │
                   user 消息注入        ← /skill-name 触发时
                   （不改 system prompt）
```

---

## 六、小结：nano → hermes 的演进

| 层 | nano | hermes |
|----|------|--------|
| **Rules** | 直接注入 `.agent/rules/*.md` | 识别标准生态文件（AGENTS.md/CLAUDE.md/.cursorrules），注入前扫描 10 种威胁模式 |
| **全局身份** | 无 | `~/.hermes/SOUL.md`，独立于项目规则 |
| **Skills 索引** | 无，靠模型记住 | system prompt 注入名称+单行描述，按需 `skill_view()` 加载完整内容 |
| **Skills 缓存** | 无 | in-process LRU + disk mtime snapshot，冷启动也快 |
| **Skills 条件** | 无 | `requires_tools`/`fallback_for_toolsets`/`platforms` 按上下文自动筛选 |
| **Skills 调用** | 直接拼到 system prompt | 注入为 user 消息，保留 prompt cache |
| **MCP 连接** | 一次性子进程 | 持久 Task + 后台事件循环 + 自动重连（指数退避） |
| **MCP 安全** | 无 | 环境变量过滤 + 错误消息凭证剥离 |
| **MCP 动态** | 无 | `tools/list_changed` 通知 → 热更新工具注册表 |

**下一篇**将解剖 **SubAgent（子智能体）**：`delegate_tool.py` 的工作原理，如何隔离上下文、防止递归、并行执行多个子任务，以及父 agent 如何只看到"任务摘要"而不是子 agent 的全部中间步骤。

---

**关键文件速查**

| 文件 | 对应内容 |
|------|---------|
| `agent/prompt_builder.py:55` | `_scan_context_content()` 上下文威胁扫描 |
| `agent/prompt_builder.py:998` | `build_context_files_prompt()` 上下文文件优先级链 |
| `agent/prompt_builder.py:575` | `build_skills_system_prompt()` 技能索引生成 |
| `agent/prompt_builder.py:441` | `_build_skills_manifest()` disk snapshot 指纹 |
| `agent/prompt_builder.py:544` | `_skill_should_show()` 条件激活逻辑 |
| `agent/skill_utils.py:92` | `skill_matches_platform()` 平台过滤 |
| `agent/skill_commands.py:121` | `_build_skill_message()` 技能注入为 user 消息 |
| `tools/mcp_tool.py:192` | `_build_safe_env()` 环境变量过滤 |
| `tools/mcp_tool.py:211` | `_sanitize_error()` 凭证剥离 |
| `tools/mcp_tool.py:720` | `MCPServerTask` 连接生命周期管理 |
| `tools/mcp_tool.py:1070` | `_mcp_loop` 后台事件循环 |
| `tools/mcp_tool.py:1125` | `_ensure_mcp_loop()` 懒启动 |
