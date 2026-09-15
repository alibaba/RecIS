# 查找对应的 debuginfo wheel

启用 `NEED_STRIP_DEBUGINFO=1` 构建后，主 wheel 的
`*.dist-info/DEBUGINFO.txt` 包含同一次构建的 debuginfo 文件名、SHA-256、
PyPI HTTP 地址及安装命令。解压主 wheel 可以直接查看此文件。

安装主 wheel 后，在同一 Python 环境运行（无需导入原生扩展）：

```bash
python -c 'from importlib.metadata import distribution; print(distribution("recis").read_text("DEBUGINFO.txt") or "此 wheel 未附带 debuginfo 安装信息")'
```

其中的 `python -m pip install "http://...whl"` 仅为示例：请按主 wheel 的安装位置，
选择对应的 Python 解释器、虚拟环境及 `--user` 等 pip 参数。
内部 PyPI 的根地址只在 `.aoneci/` 中配置，通过 `DEBUGINFO_PACKAGES_URL` 注入。
通用构建未配置该变量时，文件只提供精确文件名、校验值和本地安装示例；
开源或其他发布环境可配置自己的根地址（目录结构为发行名/版本/wheel 文件名）。
只有该构建的配对 wheel 发布后，对应下载地址才可用。
MR 或本地构建若未发布到 PyPI，请从同一流水线或本次构建产物取得文件名和
SHA-256 均匹配的 debuginfo wheel。旧版 wheel 不会自动补充此文件。
