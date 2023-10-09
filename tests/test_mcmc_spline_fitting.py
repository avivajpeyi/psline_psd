import os

import numpy as np

from slipper.sample import PsplineSampler

NTOTAL = 200


def test_simple_example(test_pdgrm: np.ndarray, tmpdir: str):
    np.random.seed(0)
    outdir = f"{tmpdir}/simple_example"
    fn = f"{outdir}/summary.png"
    PsplineSampler.fit(
        data=test_pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(Ntotal=NTOTAL, n_checkpoint_plts=5),
        spline_kwargs=dict(
            degree=3,
            k=10,
            knot_locator_type="data_peak",
        ),
    )
    assert os.path.exists(fn)
