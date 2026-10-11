"""Test NPU inference on device — run this on the actual RK3566 board."""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from core.npu_backend import InferenceBackend

def test_retinaface():
    print('=== Testing RetinaFace ===')
    rknn_path = config.RETINAFACE_RKNN
    backend = InferenceBackend(rknn_path=rknn_path)
    print(f'Backend: {backend.backend_type}')

    dummy = np.random.randint(0, 255, (1, 320, 320, 3), dtype=np.uint8)

    # Warmup
    backend.run([dummy])

    # Benchmark
    N = 20
    t0 = time.time()
    for _ in range(N):
        outputs = backend.run([dummy])
    elapsed = time.time() - t0
    fps = N / elapsed
    print(f'Output shapes: {[o.shape for o in outputs]}')
    print(f'{N} inferences in {elapsed:.2f}s = {fps:.1f} FPS')
    backend.release()

def test_face_embedder():
    print('\n=== Testing Face Embedder (w600k MobileFaceNet) ===')
    rknn_path = config.FACE_EMBED_RKNN
    backend = InferenceBackend(rknn_path=rknn_path)
    print(f'Backend: {backend.backend_type}')

    # w600k MobileFaceNet RKNN: NHWC
    dummy = np.random.randint(0, 255, (1, 112, 112, 3), dtype=np.uint8)

    backend.run([dummy])

    N = 50
    t0 = time.time()
    for _ in range(N):
        outputs = backend.run([dummy])
    elapsed = time.time() - t0
    fps = N / elapsed
    print(f'Output shapes: {[o.shape for o in outputs]}')
    print(f'Embedding dim: {outputs[0].flatten().shape[0]}')
    print(f'{N} inferences in {elapsed:.2f}s = {fps:.1f} FPS')
    backend.release()

if __name__ == '__main__':
    test_retinaface()
    test_face_embedder()
    print('\nAll tests passed.')
