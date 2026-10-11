"""ONNX -> RKNN model conversion (run on x86 with rknn-toolkit2, or on device with toolkit-lite2)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def convert_retinaface(onnx_path, rknn_path, target_platform='rk3566'):
    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    rknn.config(mean_values=[[104, 117, 123]], std_values=[[1, 1, 1]], target_platform=target_platform)
    ret = rknn.load_onnx(model=onnx_path)
    assert ret == 0, f'Load ONNX failed: {ret}'
    ret = rknn.build(do_quantization=False)
    assert ret == 0, f'Build failed: {ret}'
    ret = rknn.export_rknn(rknn_path)
    assert ret == 0, f'Export failed: {ret}'
    rknn.release()
    print(f'Converted {onnx_path} -> {rknn_path}')

def convert_mobilefacenet(onnx_path, rknn_path, target_platform='rk3566'):
    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], target_platform=target_platform)
    ret = rknn.load_onnx(model=onnx_path, inputs=['input.1'], input_size_list=[[1, 3, 112, 112]])
    assert ret == 0, f'Load ONNX failed: {ret}'
    ret = rknn.build(do_quantization=False)
    assert ret == 0, f'Build failed: {ret}'
    ret = rknn.export_rknn(rknn_path)
    assert ret == 0, f'Export failed: {ret}'
    rknn.release()
    print(f'Converted {onnx_path} -> {rknn_path}')

if __name__ == '__main__':
    models_dir = os.path.dirname(os.path.abspath(__file__))
    platform = sys.argv[1] if len(sys.argv) > 1 else 'rk3566'

    rf_onnx = os.path.join(models_dir, 'RetinaFace_mobile320.onnx')
    rf_rknn = os.path.join(models_dir, 'RetinaFace_mobile320.rknn')
    if os.path.exists(rf_onnx):
        convert_retinaface(rf_onnx, rf_rknn, platform)

    mf_onnx = os.path.join(models_dir, 'MobileFaceNet.onnx')
    mf_rknn = os.path.join(models_dir, 'MobileFaceNet.rknn')
    if os.path.exists(mf_onnx):
        convert_mobilefacenet(mf_onnx, mf_rknn, platform)
