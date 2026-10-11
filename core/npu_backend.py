"""RKNN NPU inference backend (RKNNLite only — no CPU fallback)."""
import logging

log = logging.getLogger(__name__)

class InferenceBackend:
    def __init__(self, rknn_path):
        """初始化 RKNNLite 推理后端。

        Args:
            rknn_path: .rknn 模型文件路径（必须存在）

        Raises:
            RuntimeError: RKNNLite 初始化失败
        """
        self._session = None
        self._type = 'rknn'

        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f'RKNNLite load_rknn failed: ret={ret} ({rknn_path})')
        ret = rknn.init_runtime()
        if ret != 0:
            rknn.release()
            raise RuntimeError(f'RKNNLite init_runtime failed: ret={ret}')
        self._session = rknn
        log.info(f'RKNNLite backend ready: {rknn_path}')

    @property
    def backend_type(self):
        return self._type

    def run(self, inputs):
        """Run NPU inference. inputs: list of numpy arrays. Returns list of numpy arrays."""
        return self._session.inference(inputs=inputs if isinstance(inputs, list) else [inputs])

    def release(self):
        if self._session:
            self._session.release()
            self._session = None
