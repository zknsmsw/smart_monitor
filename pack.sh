#!/bin/bash
# 打包 smart_monitor 用于部署到 oec-turbo 盒子
set -e
cd "$(dirname "$0")"

DIST_DIR="dist"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/smart_monitor/wheels"

# 复制项目文件（排除开发用的东西）
rsync -a --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='data/' \
         --exclude='dist/' \
         --exclude='venv/' \
         --exclude='*.onnx' \
         --exclude='requirements-dev.txt' \
         . "$DIST_DIR/smart_monitor/"

# 复制 rknn-toolkit-lite2 wheels（所有 Python 版本，盒子上不确定是哪个）
LITE2_DIR="/home/ecs-user/workspace/rknn-toolkit2/rknn-toolkit-lite2/packages"
if [ -d "$LITE2_DIR" ]; then
    cp "$LITE2_DIR"/rknn_toolkit_lite2-*.whl "$DIST_DIR/smart_monitor/wheels/"
    echo "Bundled rknn-toolkit-lite2 wheels"
fi

# 打包
cd "$DIST_DIR"
tar czf smart_monitor.tar.gz smart_monitor/
mv smart_monitor.tar.gz ../
cd ..
rm -rf "$DIST_DIR"

SIZE=$(du -h smart_monitor.tar.gz | cut -f1)
echo ""
echo "=== 打包完成: smart_monitor.tar.gz ($SIZE) ==="
echo ""
echo "部署步骤:"
echo "  1. scp smart_monitor.tar.gz root@<盒子IP>:/root/"
echo "  2. ssh root@<盒子IP>"
echo "  3. tar xzf smart_monitor.tar.gz && cd smart_monitor"
echo "  4. bash setup.sh"
echo "  5. source venv/bin/activate && python3 main.py"
