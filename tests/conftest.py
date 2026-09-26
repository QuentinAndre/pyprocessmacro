import matplotlib

matplotlib.use("Agg")  # plotting tests must never open a window

import numpy as np
import pandas as pd
import pytest

from pyprocessmacro import Process

BOOT = 200


@pytest.fixture(scope="session")
def data():
    """Synthetic data with a column for every symbol a model can ask for, a binary and a count outcome, a control."""
    rng = np.random.default_rng(20260925)
    n = 300
    effort, motiv, skill, value, qual, ctrl = (rng.normal(size=n) for _ in range(6))
    med1 = 0.5 * effort + 0.3 * motiv + 0.2 * effort * motiv + rng.normal(size=n)
    med2 = 0.4 * effort + rng.normal(size=n)
    outcome = (
        0.4 * med1 + 0.2 * med2 + 0.3 * effort + 0.2 * effort * skill + 0.1 * med1 * value + rng.normal(size=n)
    )
    binary = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * med1 + 0.5 * effort)))).astype(int)
    mu = np.exp(0.3 + 0.4 * med1 + 0.3 * effort)
    count = rng.negative_binomial(2, 2 / (2 + mu))  # NB2 with alpha = 0.5
    return pd.DataFrame(
        dict(effort=effort, motiv=motiv, skill=skill, value=value, qual=qual, ctrl=ctrl,
             med1=med1, med2=med2, outcome=outcome, binary=binary, count=count)
    )


@pytest.fixture
def fit(data):
    """Fit a model on the synthetic data with a small bootstrap and no init banner."""

    def _fit(model, df=None, **kwargs):
        kwargs.setdefault("boot", BOOT)
        kwargs.setdefault("seed", 1)
        kwargs.setdefault("suppr_init", True)
        return Process(data if df is None else df, model, **kwargs)

    return _fit
