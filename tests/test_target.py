import numpy as np
import slicersim

def test_supernovae():
    """ """
    snia = slicersim.LazuliSupernova(redshift=1)

    # there is no noise included, so spec must be greater or equal to zero
    lbda, spec, var = snia.get_spectrum(unit="flambda", incl_error=False)
    assert lbda.shape == spec.shape == var.shape
    assert (spec >= 0).all()

    # z=0.5 target
    snia.simulation.update(redshift=0.5)
    _ = snia.setup_to_snr(20)
    exptime = snia.get_exposure_time()
    assert exptime >= 1, "exposure shorter than 1s doesn't seem correct"

    snia.simulation.update(redshift=1)
    _ = snia.setup_to_snr(20)
    exptime_distance = snia.get_exposure_time()
    assert exptime_distance > exptime


def test_target_and_lazuli():
    """ """
    lbda = np.linspace(3_000, 20_000, 500)
    flux = np.ones( lbda.shape )*1e-18
    
    target = slicersim.LazuliTarget(lbda, flux)
    
    # config
    config = target.get_readout_config()
    assert ("nmd" in config) & ("nramps" in config)

    # sampling
    name, _ = target.get_spectrograph_sampling()
    assert name in ["fine", "medium"]

    target.change_spectrograph(sampling="fine")
    name, _ = target.get_spectrograph_sampling()
    assert name == "fine"

    target.change_spectrograph(sampling="medium")
    name, _ = target.get_spectrograph_sampling()
    assert name == "medium"

    # variance
    variance_df = target.get_variance_contribution()
    assert variance_df.shape[0] == len(target.simulation.spectrograph.lbda)
    assert np.all([k in variance_df.columns for k in target.simulation.variance_sources])

    # data volume
    data_vol = target.get_data_volume("MB")
    assert data_vol > 1, "data volume lower than 1MB, strange"    
