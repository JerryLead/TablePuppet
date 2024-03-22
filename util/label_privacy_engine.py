import torch
from typing import Any, Optional, List
from opacus.accountants.rdp import privacy_analysis
from loguru import logger


EPS = 1e-10
ROOT2 = 2.0**0.5


class RandomizedLabelPrivacy:
    def __init__(
        self,
        sigma: float,
        delta: float = 1e-10,
        mechanism: str = "Laplace",
        device: Any = None,
        seed: Optional[int] = None,
    ):
        r"""
        A privacy engine for randomizing labels.

        Arguments
            mechanism: type of the mechansim, for now either normal or laplacian
        """
        self.sigma = sigma
        self.delta = delta
        assert mechanism.lower() in ("gaussian", "laplace")
        self.isNormal = mechanism.lower() == "gaussian"  # else is laplace
        self.seed = (
            seed if seed is not None else (torch.randint(0, 255, (1,)).item())
        )  # this is not secure but ok for experiments
        self.device = device
        self.randomizer = torch.Generator(device) if self.sigma > 0 else None
        self.reset_randomizer()
        self.step: int = 0
        self.eps: float = float("inf")
        self.alphas: List[float] = [i / 10.0 for i in range(11, 1000)]
        self.alpha = float("inf")
        self.train()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def reset_randomizer(self):
        if self.randomizer is not None:
            self.randomizer.manual_seed(self.seed)

    def increase_budget(self, step: int = 1):
        if self.sigma <= 0 or step <= 0:
            return

        self.step += step
        if self.isNormal:
            rdps = privacy_analysis.compute_rdp(
                1.0, self.sigma / ROOT2, self.step, self.alphas
            )
            self.eps, self.alpha = privacy_analysis.get_privacy_spent(
                self.alphas, rdps, self.delta
            )
        else:
            if self.step > 1:
                logger.warning(
                    "It is not optimal to use multiple steps with Laplace mechanism"
                )
                self.eps *= self.step
            else:
                self.eps = 2 * ROOT2 / self.sigma

    def noise(self, shape):
        if not self._train or self.randomizer is None:
            return None
        noise = torch.zeros(shape, device=self.device)
        if self.isNormal:
            noise.normal_(0, self.sigma, generator=self.randomizer)
        else:  # is Laplace
            tmp = noise.clone()
            noise.exponential_(ROOT2 / self.sigma, generator=self.randomizer)
            tmp.exponential_(ROOT2 / self.sigma, generator=self.randomizer)
            noise = noise - tmp
        return noise

    @property
    def privacy(self):
        return self.eps, self.alpha
