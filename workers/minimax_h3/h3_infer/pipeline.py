"""MiniMax H3 T2V pipeline load + generate (diffusers ModularPipeline)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from h3_infer.request import T2VRequest

logger = logging.getLogger("minimax_h3.pipeline")

NUM_INFERENCE_STEPS = 50
MODEL_ID = "MiniMaxAI/MiniMax-H3"
DIFFUSERS_GIT_SHA = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
DEFAULT_MODEL_DIR = "/runpod-volume/minimax_h3"
FPS = 24


class PipelineError(RuntimeError):
    """Weight layout or load failure."""


class H3Pipeline:
    """Load ModularPipeline once; generate T2V MP4 to a path."""

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
        self._pipe = None
        self._load()

    def _load(self) -> None:
        root = Path(self.model_dir)
        index = root / "modular_model_index.json"
        transformer = root / "transformer"
        if not index.is_file():
            raise PipelineError(
                f"MODEL_DIR missing modular_model_index.json: {index} "
                f"(run download_weights.py --output {self.model_dir})"
            )
        if not transformer.is_dir():
            raise PipelineError(
                f"MODEL_DIR missing transformer/: {transformer} "
                f"(run download_weights.py --output {self.model_dir})"
            )

        import torch
        from diffusers.modular_pipelines import ModularPipeline

        logger.info(
            "Loading MiniMax-H3 ModularPipeline from %s (diffusers pin %s)",
            self.model_dir,
            DIFFUSERS_GIT_SHA,
        )
        local_only = os.environ.get("LOCAL_FILES_ONLY", "").lower() in {
            "1",
            "true",
            "yes",
        }
        pipe = ModularPipeline.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        pipe.load_components(dtype=torch.bfloat16)
        # 1×80GB path: auto CPU offload with reserve for activations.
        if hasattr(pipe, "components") and hasattr(
            pipe.components, "enable_auto_cpu_offload"
        ):
            pipe.components.enable_auto_cpu_offload(
                device="cuda",
                memory_reserve_margin="12GB",
            )
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        self._pipe = pipe
        logger.info("MiniMax-H3 pipeline ready")

    def generate_t2v(self, req: T2VRequest, output_path: str | Path) -> Path:
        """Run t2va inference and mux video+audio to ``output_path`` (mp4)."""
        import torch
        from diffusers.utils.export_utils import encode_video

        if self._pipe is None:
            raise PipelineError("pipeline not loaded")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        generator = torch.Generator(device="cpu").manual_seed(int(req.seed))
        state = self._pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_frames=req.length,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        )
        videos = state.get("videos")
        audio = state.get("audio")
        sampling_rate = state.get("sampling_rate")
        if not videos:
            raise PipelineError("pipeline returned no videos")

        encode_video(
            videos[0],
            fps=FPS,
            output_path=str(out),
            audio=audio[0] if audio else None,
            audio_sample_rate=sampling_rate,
        )
        if not out.is_file() or out.stat().st_size == 0:
            raise PipelineError(f"encode_video produced empty file: {out}")
        return out
