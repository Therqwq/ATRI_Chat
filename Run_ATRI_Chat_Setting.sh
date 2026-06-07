# ATRI_Chat_Setting
# 版本：v0.1.1
# 兼容平台：Linux

# 获取程序所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/venv"

# 如无法检测到配置文件"config.json"而运行失败，请尝试打开ARTI_Chat的目录后执行

# 检查虚拟环境是否存在
if [ ! -d "$VENV_DIR" ]; then
    echo "❌错误| 未找到虚拟环境目录 $VENV_DIR"
    echo "✅信息| 请先安装所需要的环境"
    exit 1
fi

# 动态获取Python版本目录
PYTHON_LIB_DIR=$(find "$VENV_DIR/lib" -maxdepth 1 -type d -name "python3.*" | head -n 1)

if [ -z "$PYTHON_LIB_DIR" ]; then
    echo "❌错误| 无法在虚拟环境中找到 Python 库目录"
    exit 1
fi

# 设置虚拟Qt插件路径，避免固执寻找系统内路径
export QT_QPA_PLATFORM_PLUGIN_PATH="$PYTHON_LIB_DIR/site-packages/PyQt5/Qt5/plugins"
export QT_QPA_PLATFORM=wayland

# 激活虚拟环境
echo "✅信息| 正在激活虚拟环境……"
# 使用bash
source "$VENV_DIR/bin/activate" || {
    echo "❌错误| 无法激活虚拟环境，请检查路径 $VENV_DIR/bin/activate"
    exit 1
}

echo "✅信息| 虚拟环境已激活：$VIRTUAL_ENV"
echo "✅信息| Qt 插件路径已设置为：$QT_QPA_PLATFORM_PLUGIN_PATH"
echo "✅信息| 正在启动ATRI_Chat_Setting……"

# 运行 Python 程序
python "$PROJECT_DIR/ATRI_Chat_Setting.py"

