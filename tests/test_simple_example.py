import os

import numpy as np

from pspline_psd.sample.gibbs_pspline_simple import gibbs_pspline_simple


def test_simple_example(helpers):
    np.random.seed(0)
    data = helpers.load_raw_data()
    data = data - data.mean()

    fn = f"{helpers.OUTDIR}/sample_metadata.png"
    gibbs_pspline_simple(
        data=data,
        Ntotal=3000,
        burnin=1000,
        degree=3,
        eqSpacedKnots=True,
        compute_psds=True,
        metadata_plotfn=fn,
        k=30,
    )
    assert os.path.exists(fn)
