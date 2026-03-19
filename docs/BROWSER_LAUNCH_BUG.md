# Why `cmd.exe /c start` Failed to Open Chrome — A Shell Context Bug

## What Happened

Three different approaches were tried to open `localhost:8501` and `localhost:8502` in Chrome on the local machine. Only the third worked.

### Attempt 1: `cmd.exe /c start <url>` — FAILED

```bash
cmd.exe /c start http://localhost:8501
```

- **Exit code**: 0 (appeared to succeed)
- **Result**: Nothing visible happened. No Chrome window appeared.

### Attempt 2: `cmd.exe /c "start chrome <url>"` — FAILED

```bash
cmd.exe /c "start chrome http://localhost:8501"
```

- **Exit code**: 0 (appeared to succeed)
- **Result**: Nothing visible happened. No Chrome window appeared.

### Attempt 3: Direct chrome.exe invocation — WORKED

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" http://localhost:8501 http://localhost:8502 &
```

- **Exit code**: 0
- **Result**: Chrome opened with both tabs showing the Streamlit apps.

---

## Why the First Two Failed

### The Execution Context

Claude Code CLI runs inside **Git Bash (MINGW64)** — a Unix-like shell environment on Windows. This creates a layered execution context:

```
┌─────────────────────────────────────────┐
│ Git Bash (MINGW64)                      │
│ Shell: /usr/bin/bash                    │
│ Environment: Unix-like paths, POSIX     │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │ cmd.exe /c start <url>          │   │
│   │ Subprocess: Windows cmd.exe     │   │
│   │ Context: inherited from Git Bash│   │
│   │ Session: possibly non-interactive│  │
│   └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

When Claude Code CLI runs `cmd.exe /c start http://localhost:8501`, here's what happens:

1. **Git Bash** spawns `cmd.exe` as a child process
2. **cmd.exe** runs the `start` command
3. **`start`** tries to open the URL using Windows' **ShellExecute** API
4. **ShellExecute** looks up the default handler for `http://` URLs
5. The handler should launch Chrome — but it may fail silently in this context

### Why `start` Fails Silently

The Windows `start` command uses `ShellExecute` internally, which is sensitive to the **execution context**:

| Factor | Normal terminal | Git Bash subprocess |
|--------|----------------|---------------------|
| **Desktop session** | Attached to user's desktop | May not be attached to interactive desktop |
| **Environment variables** | Full Windows PATH | MINGW64-modified PATH |
| **Working directory** | Standard Windows path | `/c/Users/...` (Unix-style) |
| **Shell context** | Interactive | Non-interactive subprocess |
| **Window station** | User's window station | May inherit a different window station |

The most likely causes:

#### 1. Non-Interactive Session Context

Claude Code CLI's Bash tool runs commands in a non-interactive shell. When `cmd.exe` is spawned from this context, `start` may execute the URL handler in a **background window station** — meaning the browser launches but is not attached to the user's visible desktop. It's running, but you can't see it.

#### 2. Window Station Isolation

Windows has a concept of **window stations** — isolated environments for desktop rendering. Processes in different window stations can't display windows on each other's desktops. If `cmd.exe` inherits a non-default window station from the Git Bash process, `start` may launch Chrome in an invisible window station.

#### 3. URL Handler Indirection

The `start` command with a URL goes through multiple layers:

```
start http://localhost:8501
  → ShellExecute("open", "http://localhost:8501")
    → Registry lookup: HKEY_CLASSES_ROOT\http\shell\open\command
      → Chrome launcher (possibly via ProgId redirect)
        → chrome.exe --single-argument http://localhost:8501
```

Each layer of indirection is a point where the execution context (environment, PATH, window station) can cause silent failure. The process may start but be invisible, or the handler lookup may fail and `start` still returns exit code 0.

#### 4. PATH Conflicts

Git Bash modifies the PATH significantly. It prepends `/mingw64/bin`, `/usr/bin`, etc. When `cmd.exe` inherits this PATH, Windows program lookups can behave unexpectedly. The `start chrome` variant (Attempt 2) tries to find `chrome` as a program name, but the MINGW64 PATH doesn't include Chrome's directory.

---

## Why Direct Invocation Worked

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" http://localhost:8501 &
```

This bypasses all the indirection:

| Layer | `cmd.exe /c start` | Direct `chrome.exe` |
|-------|-------------------|---------------------|
| Shell | Git Bash → cmd.exe → start → ShellExecute → handler → Chrome | Git Bash → Chrome |
| Indirection levels | 5+ | 1 |
| URL handler lookup | Yes (can fail) | No (bypassed) |
| Window station | Inherited through cmd.exe (may be wrong) | Inherited directly from Git Bash (correct) |
| PATH dependency | Yes (`start chrome` needs Chrome in PATH) | No (absolute path used) |
| Exit code on failure | 0 (silent) | Non-zero (visible) |

By using the **absolute path** to `chrome.exe` and passing URLs as arguments directly, we skip:
- The `cmd.exe` subprocess
- The `start` command
- The `ShellExecute` API
- The URL protocol handler registry lookup
- Any PATH-based program resolution

Git Bash can directly execute Windows `.exe` files. The `&` puts it in the background so it doesn't block the shell. Chrome receives the URLs as command-line arguments and opens them as tabs.

---

## The Silent Failure Problem

The most insidious aspect: **`start` returned exit code 0 in all cases.** It reported success even when nothing visible happened.

This is because `start` considers its job done when it successfully calls `ShellExecute` — regardless of whether the target application actually appeared on screen. The URL handler may have:
- Started Chrome in an invisible window station
- Queued the URL for a Chrome instance that couldn't render
- Launched a handler process that immediately exited
- Completed the registry lookup but failed at the launch step

None of these register as errors to `start`.

```
cmd.exe /c start http://localhost:8501
                │
                ├── ShellExecute called successfully → exit code 0
                │
                └── Chrome launched in background/invisible
                    └── User sees nothing
                    └── Claude thinks it worked
```

---

## Reproducing the Issue

This bug is specific to running `start` from a non-interactive subprocess spawned by Git Bash / MINGW64. You can verify:

```bash
# From a normal cmd.exe or PowerShell terminal (interactive):
start http://localhost:8501
# → Works fine, Chrome opens

# From Git Bash directly:
cmd.exe /c start http://localhost:8501
# → May or may not work depending on session context

# From a subprocess of Git Bash (how Claude Code runs commands):
bash -c 'cmd.exe /c start http://localhost:8501'
# → Likely fails silently (same bug)
```

---

## The Fix: Always Use Direct Invocation

For Claude Code CLI on Windows (Git Bash), always open browsers by invoking the executable directly:

```bash
# Chrome (absolute path)
"/c/Program Files/Google/Chrome/Application/chrome.exe" http://localhost:8501 &

# Edge
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" http://localhost:8501 &

# Firefox
"/c/Program Files/Mozilla Firefox/firefox.exe" http://localhost:8501 &
```

If you don't know the browser path, find it first:

```bash
# Find Chrome
ls "/c/Program Files/Google/Chrome/Application/chrome.exe" 2>/dev/null

# Find Edge
ls "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" 2>/dev/null
```

---

## Summary

| Approach | Works? | Why |
|----------|--------|-----|
| `cmd.exe /c start http://...` | No | ShellExecute may launch Chrome in invisible window station from non-interactive Git Bash subprocess |
| `cmd.exe /c "start chrome http://..."` | No | Same window station issue + `chrome` not in MINGW64 PATH |
| `chrome.exe http://... &` (absolute path) | Yes | Direct execution, no indirection, no ShellExecute, no PATH lookup |

**Root cause**: The `start` command relies on `ShellExecute`, which is sensitive to the desktop session and window station context. When called from a non-interactive Git Bash subprocess (how Claude Code CLI runs Bash commands), it can launch the browser in an invisible context. Direct invocation of the browser executable bypasses all these layers and reliably opens visible windows.
