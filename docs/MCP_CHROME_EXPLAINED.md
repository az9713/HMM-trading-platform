# Why Chrome Opened on the Wrong Computer — MCP Architecture Explained

## What Happened

During this session, the user asked Claude Code CLI to open two browser tabs showing the original and redesigned Streamlit dashboards (`localhost:8501` and `localhost:8502`).

Claude Code used the `mcp__claude-in-chrome__navigate` tool to open the URLs. The tabs opened in Chrome — but on **a different computer** (the one running Claude Desktop), not on the local machine running Claude Code CLI where the Streamlit servers were actually running.

**The twist**: Both machines had Chrome installed with the "Claude in Chrome" MCP extension. The extension existed locally, but MCP routed the commands to the remote machine's Chrome anyway.

The fix was trivial — use `cmd.exe /c start http://localhost:8501` to open URLs via the local OS. But the failure reveals something important about how MCP tool routing actually works.

---

## The Setup (Two Machines, Both with Chrome Extension)

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│         MACHINE A               │    │         MACHINE B               │
│    (Claude Code CLI machine)    │    │   (Claude Desktop machine)      │
│                                 │    │                                 │
│  ┌───────────────────────┐      │    │  ┌───────────────────────┐      │
│  │ Claude Code CLI       │      │    │  │ Claude Desktop        │      │
│  │ (terminal session)    │      │    │  │ (desktop app)         │      │
│  └───────────┬───────────┘      │    │  └───────────────────────┘      │
│              │                  │    │            │                    │
│  ┌───────────▼───────────┐      │    │  ┌─────────▼─────────────┐      │
│  │ Streamlit app.py      │      │    │  │ Chrome Browser        │      │
│  │ localhost:8501 (main)  │      │    │  │ + Claude-in-Chrome    │      │
│  │ localhost:8502 (wktree)│      │    │  │   MCP extension       │      │
│  └───────────────────────┘      │    │  │   ← MCP connected ─┐  │      │
│                                 │    │  └─────────────────────│──┘      │
│  ┌───────────────────────┐      │    │                        │         │
│  │ Chrome Browser        │      │    │  MCP commands go here  │         │
│  │ + Claude-in-Chrome    │      │    │  because this Chrome   │         │
│  │   MCP extension       │      │    │  has the active MCP    │         │
│  │   ← NOT connected     │      │    │  session               │         │
│  └───────────────────────┘      │    └────────────────────────│─────────┘
│                                 │                             │
│  Extension is installed but     │           ┌────────────────┘
│  NOT the one handling MCP       │           │
│  commands for this session      │           ▼
│                                 │    Anthropic's API servers
└─────────────────────────────────┘    (route MCP calls to the
                                        registered MCP server)
```

Both machines have Chrome with the extension. But only **one** has an active MCP connection for this session.

---

## Why the Wrong Chrome Answered

### How the Chrome MCP Extension Works

The "Claude in Chrome" extension is an **MCP server** — it registers itself as a tool provider that Claude can call. But it doesn't just magically connect to any Claude instance. It has to establish a session with a specific Claude client.

Here's the connection flow:

```
Chrome Extension (MCP Server)
        │
        │ registers via websocket/native messaging
        ▼
Claude Client (MCP Host)
        │
        │ exposes tools to the model
        ▼
Claude (Anthropic API)
        │
        │ calls mcp__claude-in-chrome__* tools
        ▼
Routed back to the Chrome that registered
```

### The Key Question: Which Chrome Registered?

There are two Chrome browsers with the extension, but only one established the MCP connection for this conversation. The question is: **how does the extension decide which Claude client to connect to?**

The Chrome MCP extension connects to **Claude Desktop**, not Claude Code CLI. This is because:

1. **Claude Desktop acts as an MCP host natively.** It has built-in MCP server management — when you configure MCP servers in Claude Desktop's settings, it establishes persistent connections to them. The Chrome extension is designed to pair with Claude Desktop as its primary MCP host.

2. **Claude Code CLI inherits MCP tools differently.** Claude Code CLI can use MCP tools, but it discovers them through its own configuration (`~/.claude/settings.json` or project-level MCP configs). The Chrome extension on the local machine may be installed but not configured as an MCP server for Claude Code CLI.

3. **Only one active connection per extension instance.** The Chrome extension on Machine B connected to Claude Desktop on Machine B. The Chrome extension on Machine A was installed but had no active MCP session with Claude Code CLI on Machine A.

4. **Claude Code CLI got the Chrome tools from Machine B's connection.** When Claude Code CLI's conversation started, the available MCP tools included `mcp__claude-in-chrome__*`. These tools were routed through the MCP infrastructure to Machine B's Chrome — the one with the active connection.

### The Connection Path for This Session

```
User types in Claude Code CLI (Machine A)
        │
        ▼
Claude (Anthropic API)
        │
        ├── Bash, Read, Edit tools → Machine A (local to Claude Code CLI)
        │
        └── mcp__claude-in-chrome__* → Machine B's Chrome
                │
                │  WHY? Because Machine B's Chrome extension
                │  has the active MCP registration. Machine A's
                │  Chrome extension is installed but not connected
                │  to this session.
                │
                └── Chrome on Machine B navigates to localhost:8501
                    └── "localhost" = Machine B = nothing running there
```

---

## Why Having the Extension Installed Isn't Enough

Installing the Chrome extension on Machine A doesn't automatically make it the MCP server for Claude Code CLI sessions on Machine A. The extension needs to:

1. **Be configured as an MCP server** in Claude Code CLI's settings
2. **Establish a connection** with the Claude Code CLI process
3. **Be the one that registered** the `mcp__claude-in-chrome__*` tools for this specific session

Think of it like having Slack installed on two computers. Both have the app, but notifications go to whichever one you're logged into and have active. The Chrome MCP extension is similar — "installed" and "connected to this session" are two different things.

### How MCP Server Registration Works

```
MCP Server (Chrome Extension)          MCP Host (Claude Client)
        │                                       │
        │── "I provide these tools:             │
        │    navigate, tabs_create,              │
        │    get_page_text, ..."  ──────────────►│
        │                                       │
        │                                       │── registers tools
        │                                       │── makes them available
        │                                       │   to the model
        │                                       │
        │◄── "navigate to localhost:8501" ───────│
        │                                       │
        │── executes in THIS Chrome ────►        │
```

Machine B's Chrome did step 1 (registered tools). Machine A's Chrome did not register with this Claude Code CLI session — so even though the extension exists on Machine A, it's invisible to this conversation.

---

## The localhost Trap

Even if MCP had routed to Machine A's Chrome, there's a subtler issue with `localhost`:

`localhost` always means **"the machine executing this command."**

| Tool | Executes on | `localhost` resolves to |
|------|------------|----------------------|
| `Bash: curl localhost:8501` | Machine A | Machine A (correct) |
| `Bash: cmd.exe /c start localhost:8501` | Machine A | Machine A (correct) |
| `mcp__chrome__navigate(localhost:8501)` | Machine B | Machine B (wrong) |

If the Chrome extension on Machine A had been connected, it would have worked — `localhost` on Machine A's Chrome would have correctly reached Machine A's Streamlit. The problem was specifically that the wrong machine's Chrome was connected.

---

## How to Fix This

### Option 1: Use Bash to Open Local URLs (Recommended)

Don't use Chrome MCP tools for `localhost` URLs. Use the OS-level open command instead:

```bash
# Windows
cmd.exe /c start http://localhost:8501

# Mac
open http://localhost:8501

# Linux
xdg-open http://localhost:8501
```

This always runs on the Claude Code CLI machine, opening the local browser.

### Option 2: Connect the Local Chrome Extension to Claude Code CLI

Configure the Chrome MCP extension on Machine A as an MCP server for Claude Code CLI. This would make `mcp__claude-in-chrome__*` tools route to Machine A's Chrome instead of Machine B's.

The configuration would go in Claude Code CLI's MCP settings. However, the Chrome extension is primarily designed to pair with Claude Desktop, so this may require additional setup.

### Option 3: Use Network URLs Instead of localhost

When Streamlit starts, it shows both URLs:

```
Local URL:    http://localhost:8501
Network URL:  http://192.168.1.90:8501
```

Using the Network URL (`192.168.1.90:8501`) would work from any machine on the network, including Machine B's Chrome. This sidesteps the localhost problem entirely — but still opens the browser on the wrong machine.

---

## The Bigger Picture: MCP Tool Location Awareness

This incident reveals a fundamental property of MCP architecture:

**MCP tools have no location metadata.**

When Claude sees a tool like `mcp__claude-in-chrome__navigate`, it knows:
- The tool's name
- The tool's parameters (url, tabId)
- The tool's description

It does NOT know:
- Which physical machine the tool runs on
- Whether `localhost` in tool parameters will resolve to the same machine as `localhost` in Bash
- The network topology between MCP servers
- Whether two tools are co-located or on different continents

This is by design — MCP abstracts away the transport layer. A tool could run locally, on a LAN machine, or in a cloud server. The model doesn't know and isn't supposed to care. But when location matters (as with `localhost` URLs), this abstraction leaks.

### The Abstraction Leak

```
Claude's mental model:        Reality:

"I have a browser tool         Browser tool runs on Machine B
 and a bash tool.              Bash tool runs on Machine A
 Both are equally local."      localhost means different things
                               for each one.
```

---

## Summary

| Question | Answer |
|----------|--------|
| **What happened?** | Chrome MCP tools opened tabs on Machine B instead of Machine A |
| **Both machines had the extension?** | Yes, but only Machine B's extension had an active MCP session |
| **Why Machine B?** | The Chrome extension connects to Claude Desktop. Claude Desktop was on Machine B. Machine A's extension was installed but not connected to this Claude Code CLI session |
| **Is this a bug?** | No — it's how MCP routing works. Tools execute where the MCP server is registered, not where Claude Code CLI runs |
| **The localhost problem** | `localhost` resolves to whichever machine executes the command. Different MCP servers = different machines = different `localhost` |
| **The fix** | Use `cmd.exe /c start <url>` (Bash tool) for local URLs. Reserve Chrome MCP for remote web pages |
| **The lesson** | MCP tools have no location awareness. When location matters (localhost, file paths, ports), use tools that run locally (Bash, Read, Write) |
