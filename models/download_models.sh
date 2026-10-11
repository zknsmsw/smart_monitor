#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Downloading RetinaFace ONNX model..."
if [ ! -f "RetinaFace_mobile320.onnx" ]; then
    wget -q -O RetinaFace_mobile320.onnx https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/RetinaFace/RetinaFace_mobile320.onnx
    echo "Downloaded RetinaFace_mobile320.onnx"
else
    echo "RetinaFace_mobile320.onnx already exists"
fi

echo "Downloading MobileFaceNet ONNX model..."
if [ ! -f "MobileFaceNet.onnx" ]; then
    wget -q -O MobileFaceNet.onnx https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/MobileFaceNet/MobileFaceNet.onnx 2>/dev/null || \
    echo "WARNING: MobileFaceNet download failed. Please manually place MobileFaceNet.onnx in models/"
    echo "Downloaded MobileFaceNet.onnx"
else
    echo "MobileFaceNet.onnx already exists"
fi

echo "Model download complete."
