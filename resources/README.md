# 官方资源来源

- 上游仓库：<https://github.com/DimABSA/DimABSA2026>
- 本地快照对应提交：`bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`
- 下载日期：2026-08-10
- 已通过本机 `7897` HTTP 代理下载完整上游仓库；当前本地资源约 61 MB（不含上游 `.git`）。

`resources/DimABSA2026/` 已加入 `.gitignore`。官方规则限制数据重新分发；个人 GitHub 仓库应提交下载说明和代码，不要重新提交原始数据。

如果网络正常且本地还没有资源，可在项目根目录运行：

```bash
PROXY_URL=http://127.0.0.1:7897 bash scripts/download_official_resources.sh
```

不需要代理时直接省略 `PROXY_URL=...`。脚本只对当前克隆命令设置代理，不修改全局 Git 配置。
