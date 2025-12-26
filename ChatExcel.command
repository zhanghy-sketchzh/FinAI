#!/bin/bash

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到项目目录
cd "$SCRIPT_DIR" || exit

# 显示启动信息
echo "=========================================="
echo "正在启动 DB-GPT 服务..."
echo "配置文件: configs/dbgpt-proxy-siliconflow.toml"
echo "=========================================="
echo ""

# 启动服务
uv run python packages/dbgpt-app/src/dbgpt_app/dbgpt_server.py --config configs/dbgpt-proxy-siliconflow.toml

# 如果服务退出，保持终端窗口打开以便查看错误信息
if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "服务启动失败，请检查错误信息"
    echo "=========================================="
    read -p "按回车键关闭窗口..."
fi

