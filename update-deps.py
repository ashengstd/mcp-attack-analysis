import re
import subprocess
import sys
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
MODE = "ge"  # 默认 >=
if len(sys.argv) > 1 and sys.argv[1] == "--exact":
    MODE = "eq"

CUDA_PKGS = {"torch", "torchaudio", "torchvision"}
RED = "\033[0;31m"
GREEN = "\033[0;32m"
NC = "\033[0m"

# --- 获取当前环境版本 ---
subprocess.run(["uv", "sync", "-U"])
result = subprocess.run(
    ["uv", "pip", "list", "--format=freeze"], capture_output=True, text=True
)
versions = {}
for line in result.stdout.strip().splitlines():
    if "==" not in line:
        continue
    name, ver = line.split("==", 1)
    lname = name.strip().lower()
    versions[lname] = ver.strip()

# --- 读取 pyproject.toml 并更新 ---
with PYPROJECT.open("r") as f:
    lines = f.readlines()

new_lines = []
in_deps = False
dep_line_regex = re.compile(r'^(\s*)"([^"]+)"(\s*,?\s*)(\n?)$')  # 捕获原行换行符

for line in lines:
    stripped = line.strip()

    # 开始依赖列表
    if stripped.startswith("dependencies") and stripped.endswith("["):
        in_deps = True
        new_lines.append(line)
        continue

    # 结束依赖列表
    if in_deps and stripped == "]":
        in_deps = False
        new_lines.append(line)
        continue

    # --- 在依赖列表中处理每行 ---
    if in_deps:
        m = dep_line_regex.match(line)
        if m:
            indent, raw_dep, trailing, newline = m.groups()
            pkg_part = re.split(r"[~<>=!]", raw_dep)[0].strip()
            lname = pkg_part.lower()
            old_constraint = raw_dep[len(pkg_part) :]

            new_ver = versions.get(lname)
            if new_ver:
                # CUDA 特殊处理
                if lname in CUDA_PKGS:
                    clean_ver = new_ver.split("+")[0]
                    if MODE == "eq":
                        new_constraint = f"=={clean_ver}"
                    else:
                        new_constraint = f">={clean_ver}"
                    # 只有版本不同时才更新
                    if old_constraint != new_constraint:
                        new_line = (
                            f'{indent}"{pkg_part}{new_constraint}"{trailing}{newline}'
                        )
                        new_lines.append(new_line)
                        print(
                            f"🔄 {pkg_part}: {RED}{old_constraint}{NC} → {GREEN}{new_constraint}{NC} (CUDA suffix dropped)"
                        )
                    else:
                        new_lines.append(line)  # 保留原行
                    continue

                # 通用依赖
                if MODE == "eq":
                    new_constraint = f"=={new_ver}"
                else:
                    new_constraint = f">={new_ver}"
                # 只有版本不同才更新
                if old_constraint != new_constraint:
                    new_line = (
                        f'{indent}"{pkg_part}{new_constraint}"{trailing}{newline}'
                    )
                    new_lines.append(new_line)
                    if old_constraint:
                        print(
                            f"🔄 {pkg_part}: {RED}{old_constraint}{NC} → {GREEN}{new_constraint}{NC}"
                        )
                    else:
                        print(
                            f"✨ {pkg_part}: (no version) → {GREEN}{new_constraint}{NC}"
                        )
                else:
                    new_lines.append(line)  # 保留原行
                continue

    # 非依赖行或未匹配行
    new_lines.append(line)

# 写回 pyproject.toml
with PYPROJECT.open("w") as f:
    f.writelines(new_lines)

print(f"✅ {PYPROJECT} has been updated.")
