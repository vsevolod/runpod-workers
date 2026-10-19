"""Model adapters for the common image worker; constructors never load weights."""

import os
from pathlib import Path
import sys

WORKERS = Path(__file__).resolve().parents[2]
for directory in (WORKERS, WORKERS / 'krea2', WORKERS / 'qwen_image_21'):
    sys.path.insert(0, str(directory))

from shared.api import KREA_API, QWEN_API
from shared.prefetch import PrefetchSession
from krea2_infer import load_pipeline as load_krea_pipeline
from krea2_infer.api import infer as infer_krea
from krea2_infer.dit_quant import resolve_dit_quant
from krea2_infer.lora import LoRAError as KreaLoRAError
from krea2_infer.request import normalize_job_input as normalize_krea
from qwen_image_21_infer.api import infer as infer_qwen
from qwen_image_21_infer.lora import LoRAError as QwenLoRAError
from qwen_image_21_infer.model_manager import ModelManager as QwenModelManager
from qwen_image_21_infer.request import normalize_job_input as normalize_qwen


def krea_quantization():
    return resolve_dit_quant(os.environ.get('KREA_DIT_QUANT') or os.environ.get('DIT_QUANT', 'fp8'))


class KreaBackend:
    profile = KREA_API
    client_errors = (KreaLoRAError,)
    oom_message = 'CUDA out of memory. Reduce the canvas/reference budget or use a larger GPU.'

    def __init__(self):
        self.pipe = None

    @staticmethod
    def normalize(values, *, raw_keys):
        return normalize_krea(values, raw_keys=raw_keys, quant_mode=krea_quantization())

    def load(self):
        directory = Path(os.environ.get('KREA_MODEL_DIR') or os.environ.get('MODEL_DIR') or '/models/krea2')
        loras = Path(os.environ.get('KREA_LORA_DIR') or os.environ.get('LORA_DIR') or '/runpod-volume/krea2/loras')
        quantization = krea_quantization()
        command = [sys.executable, '-u', str(WORKERS / 'krea2/download_weights.py'),
                   '--output', str(directory), '--quant', quantization, '--skip-vae-file']
        with PrefetchSession(command) as pending:
            self.pipe = load_krea_pipeline(model_dir=directory, lora_dir=loras,
                                           quant_mode=quantization, local_files_only=True)
            pending.wait('all')

    def infer(self, norm, seed, raw_loras):
        return infer_krea(self.pipe, norm, seed, raw_loras)

    def close(self):
        self.pipe = None


class QwenBackend:
    profile = QWEN_API
    normalize = staticmethod(normalize_qwen)
    client_errors = (QwenLoRAError,)
    oom_message = 'CUDA out of memory. Keep OFFLOAD=cpu and reduce the canvas/reference budget.'

    def __init__(self):
        self.manager = None

    def load(self):
        self.manager = QwenModelManager()

    def infer(self, norm, seed, raw_loras):
        return infer_qwen(self.manager, norm, seed, raw_loras)

    def close(self):
        manager, self.manager = self.manager, None
        if manager is not None:
            manager.close()


REGISTRY = {'krea2': KreaBackend, 'qwen_image_21': QwenBackend}
