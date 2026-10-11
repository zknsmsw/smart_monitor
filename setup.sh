#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Smart Monitor Setup ==="

# Detect Python version
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Python: $(python3 --version), tag: $PY_VER"

# Create venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtualenv"
fi
source venv/bin/activate
pip install -q --upgrade pip

# Install base deps
pip install -q -r requirements.txt

# Install rknn-toolkit-lite2 if wheel is bundled
LITE2_WHL=$(ls wheels/rknn_toolkit_lite2-*-${PY_VER}-*.whl 2>/dev/null | head -1)
if [ -n "$LITE2_WHL" ]; then
    pip install -q "$LITE2_WHL" && echo "Installed rknn-toolkit-lite2 from bundled wheel" || echo "WARN: rknn-toolkit-lite2 install failed"
else
    echo "No bundled rknn-toolkit-lite2 wheel found for $PY_VER"
    echo "  On device: place the .whl in wheels/ and re-run, or install manually:"
    echo "  pip install rknn_toolkit_lite2-2.3.2-${PY_VER}-...-aarch64.whl"
fi

# Create data dirs
mkdir -p data/recordings data/face_crops

# ── Compile C extensions for face-detection post-processing ──────────
# Replaces pure-Python NMS / box_decode / landmark decode with C loops.
# Falls back silently to Python if compilation fails or gcc is absent.
echo ""

_compile_c_ext() {
    local src="$1"
    local so="$2"
    if [ -f "$src" ]; then
        if command -v gcc >/dev/null 2>&1; then
            if gcc -O3 -march=armv8.2-a -shared -fPIC -o "$so" "$src" -lm 2>/dev/null; then
                echo "  ✓ $so"
            elif gcc -O3 -shared -fPIC -o "$so" "$src" -lm 2>/dev/null; then
                echo "  ✓ $so (generic, no -march)"
            else
                echo "  ⚠ $src compilation failed — Python fallback"
            fi
        else
            echo "  ⚠ gcc not found — skipping C extensions (Python fallback)"
        fi
    else
        echo "  ⚠ $src not found — skipping"
    fi
}

echo "--- Compiling C extensions ---"
_compile_c_ext "core/_nms_fast.c"      "core/_nms_fast.so"
_compile_c_ext "core/_decode_landm.c"  "core/_decode_landm.so"

echo ""
echo "=== Setup complete ==="
echo "Run: source venv/bin/activate && python main.py"
