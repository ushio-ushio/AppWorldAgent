# SIWC 手动浏览器登录（Windows）

当自动打开浏览器会新开窗口、切换配置文件或导致现有浏览器会话重置时，使用本流程。授权页由用户粘贴到当前已登录的 Edge 标签页；项目不自动启动浏览器。

## 已验证流程

验证日期：2026-11-24。SIWC 登录成功采用以下方式：

1. 使用 Windows Conda `agent` 解释器和当前 Windows 用户启动 SIWC 登录回调服务。
2. 回调服务绑定临时 `127.0.0.1` 端口，并在等待状态下输出一次性授权链接。
3. 不调用 `webbrowser.open()`，不使用 `msedge --new-window`，不新建浏览器配置文件。
4. 用户将授权链接复制到原有 Edge 窗口的地址栏，完成 ChatGPT 登录和授权。
5. 浏览器回跳到本机 `127.0.0.1` 后，登录进程校验 `state`、PKCE、ID Token 和授权 scope，并将凭据保存到 Windows DPAPI 存储。

授权链接含临时 OAuth 状态，只用于当前登录尝试。不要把链接、回调 URL、验证码或令牌写入日志、报告或仓库。手动模式省略可选的 `id_token_hint`，因此可能显示账号选择页；这是预期行为。

## 启动

必须在能被桌面 Edge 访问其本机回环端口的 **Windows 主机 shell** 中运行，不要在隔离容器、远程 Linux 或无法共享 Windows `127.0.0.1` 的执行沙箱中启动监听器。建议使用 Windows Terminal / PowerShell，并确保使用同一 Windows 用户，以便 DPAPI 凭据可用：

```powershell
$repo = 'D:\worldapp\.v1_2_b_stage'
& "$env:CONDA_PREFIX\python.exe" "$repo\scripts\siwc_manual_login.py"
```

程序显示授权链接后，将其复制到原有 Edge 标签页地址栏并完成授权。保持 PowerShell 进程运行，直到输出 `SIWC sign-in completed.`。默认等待 10 分钟；超时后旧链接和回调状态失效，需要重新运行脚本生成新的链接。

## 故障判断

- 授权域名打不开：检查当前浏览器的网络/代理连接；不要先改 AppWorld 代码。
- 页面能授权但 `127.0.0.1` 回跳超时：确认 PowerShell 登录进程仍在运行，并且是在 Windows 主机用户上下文中启动，而不是隔离执行环境。
- 回跳成功但 scope 不完整、账号不匹配或 token exchange 失败：保留错误类别，不要记录授权码或令牌；重新运行会生成新的 `state`、nonce、PKCE 和回调端口。
- 不要重放旧授权链接，也不要把 API Key、Codex CLI 凭据或 SIWC token 复制到日志中。

## 实现入口

- `scripts/siwc_manual_login.py`：手动粘贴授权链接的启动入口。
- `src/agentlab/models/siwc_auth.py::login(open_browser=False)`：不启动浏览器，只输出授权链接并等待本机回调；仍执行原有 OAuth 校验和 DPAPI 保存逻辑。
