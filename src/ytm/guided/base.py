from typing import Any, Unpack, cast

import numpy as np

from .._core.base import BaseTM as CoreBaseTM
from .._core.device_config import DeviceConfig, split_device_kwargs
from .config import T_Config, TMConfig


class BaseTM(CoreBaseTM):
    @staticmethod
    def _config_cls() -> type[TMConfig]:
        return TMConfig

    @staticmethod
    def _device_cls(kind: str):
        if kind == "cpu":
            from .backends.cpu import CPUDevice

            return CPUDevice
        from .backends.cuda import CUDADevice

        return CUDADevice

    def __init__(
        self,
        n_clauses: int,
        s: float,
        dim: int | tuple[int, ...],
        n_classes: int,
        **opt: Unpack[T_Config],
    ):
        cfg_kw, dev_kw = split_device_kwargs(dict(opt))
        config = self._config_cls()(n_clauses=n_clauses, s=s, dim=dim, n_classes=n_classes, **cfg_kw)
        super().__init__(config, DeviceConfig(**dev_kw))

    @property
    def config(self) -> TMConfig:
        return cast(TMConfig, self.dev.config)

    def pre_epoch(self) -> None:
        self._loss_sum: Any = 0.0

    def post_sample(self, e: int) -> None:
        self._loss_sum += self.dev.buf.loss[0].astype(np.float64)

    def post_epoch(self) -> float:
        return float(self._loss_sum) / self.dev.buf.N

    def to_prob(self, class_sums: np.ndarray) -> np.ndarray:
        """No-op, simply returns the activated class sums (``class_sums``) as is."""
        return class_sums
