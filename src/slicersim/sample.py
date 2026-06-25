import numpy as np
import pandas
from astropy import units as u

def broadcast_mapping(value, ntargets):
    """Broadcast a value to a given number of targets."""
    value = np.atleast_1d(value)
    if np.ndim(value)>1:
        # squeeze drop useless dimensions.
        broadcasted_values = np.broadcast_to(value, (ntargets, value.shape[-1]) )
    else:
        broadcasted_values = np.broadcast_to(value, ntargets)

    return broadcasted_values

def build_exptime_estimator(snr, per_resolution=True,
                                redshift_range=[0.01, 1.5], ntrails=20,
                                **kwargs):
    """
    Parameters
    ----------

    Returns
    -------
    Univariate spline
        interpolate exptime [in second] given redshift.
    """
    from scipy.interpolate import make_smoothing_spline

    # build a reference sample
    dset = Sample.from_sneia( np.linspace(*redshift_range, ntrails) )

    # setup to the requested snr
    _ = dset.setup_to_snr(snr, per_resolution=per_resolution, **kwargs)

    # get the summary statistics from which the interpolator will be built
    stats = dset.get_summary_stats()
    stats["redshift"] = dset.call_down("get_properties", 'redshift')

    return make_smoothing_spline(*stats.sort_values("redshift")[["redshift", "exptime"]].values.T)

class Sample( object ):

    def __init__(self, targets, setup_spectrograph=True):
        """ """
        self._targets = targets
        if setup_spectrograph:
            self.setup_spectrograph()

    @classmethod
    def from_sneia(cls, redshifts, colors=0, stretchs=0, **kwargs):
        """ """
        from . import LazuliSupernova
        data = pandas.DataFrame({"redshift": redshifts,
                                 "stretch": stretchs,
                                 "color": colors})
        targets = [LazuliSupernova(**(this.to_dict() | kwargs)) for index, this in data.iterrows()]
        return cls(targets)

    # ========== #
    #  methods   #
    # ========== #
    # -------- #
    #  GETTER  #
    # -------- #
    def get_target(self, index):
        """ """
        return self.targets[index]

    def get_exposure_time(self, units="second", full=False, **kwargs):
        """ """
        return np.asarray( self.call_down("get_exposure_time", full=full, **kwargs)) * u.second.to(units)

    def get_data_volume(self, units="GB"):
        """ """
        return np.asarray( self.call_down("get_data_volume") )

    def get_readout_config(self):
        """ """
        readout = pandas.DataFrame(self.call_down("get_readout_config"))
        nmd = readout.pop("nmd")
        return readout.join(pandas.DataFrame(np.stack(nmd.values),
                                             columns=["ngroups", "nframes_per_group", "ndrops"])
                           )

    def get_summary_stats(self):
        """ """
        data_volume = self.get_data_volume()
        exposure_time = self.get_exposure_time("s")
        readout_config = self.get_readout_config()

        return pandas.DataFrame({"volume":data_volume,
                                  "exptime":exposure_time,
                                 }).join(readout_config)

    def get_total_surveyduration(self, sample_size=8000, fraction_survey=0.4, units="yr"):
        """ """
        return self.get_exposure_time(units).sum() * sample_size/self.ntargets / fraction_survey

    # -------- #
    #  SETUP   #
    # -------- #
    def change_property(self, **kwargs):
        """ """
        return [target.simulation.update(**kwargs) for target in self.targets]


    # -------- #
    #  SETUP   #
    # -------- #
    def setup_spectrograph(self, redshift_cut=0.8):
        """ This is setup the spectrograph mode of each target,
        fine for these with redshift<redshift_cut, medium otherwise """
        for target in self.targets:
            redshift = target.get_properties("redshift")
            if redshift <= redshift_cut:
                target.change_spectrograph("narrow")
            else:
                target.change_spectrograph("wide")

    def setup_to_snr(self, snr, per_resolution=True,
                     lbda_range=[4000, 6800], frame='rest',
                     statistic=np.mean, inplace=True,
                     client=None, show_progress=False,
                     **kwargs):
        """ setup all individual targets to this snr """


        prop = dict(per_resolution=per_resolution,
                    lbda_range=lbda_range, frame=frame,
                    statistic=statistic, inplace=inplace
                    ) | kwargs

        # no dask client. Fine, let's use call_down
        if client is None:
            configs_and_snr = self.call_down("setup_to_snr", snr=snr, show_progress=show_progress,
                                                 **prop)
            configs = [case_[0] for case_ in configs_and_snr]
            snrs = [case_[1] for case_ in configs_and_snr]

        else:
            # dask client given
            import dask
            configs, snrs = [], []
            for target in self.targets:
                # dask will not manage inplace=True. This has to be done manually if needed
                thisout = dask.delayed(target.setup_to_snr)(snr, **(prop| {"inplace":False}) ).persist()
                configs.append(thisout[0])
                snrs.append(thisout[1])

            # => 'thisout' has been persisted, so this should be quick.
            configs = client.gather( client.compute(configs) )

            # if inplace needed, this is now where it's done,
            # now that each individual configs are known.
            if inplace:
                for config, target in zip(configs, self.targets):
                    target.change_detector(**config)

            snrs = client.gather(client.compute(snrs))

        # get as dataframe and array
        return pandas.DataFrame(configs), np.stack(snrs)

    # ============= #
    #  Internal     #
    # ============= #
    def call_down(self, which, mapargs=None, allow_call=True, level_down="",
                      show_progress=False, **kwargs):
        """ Call a method on each target in the collection. """
        if show_progress:
            from tqdm import tqdm
        # applied to target.simulation
        if level_down is not None:
            level_down = level_down.strip()
            if level_down != "" and not level_down.startswith("."):
                level_down = f".{level_down}"

        if mapargs is not None:
            mapargs = broadcast_mapping(mapargs, self.ntargets)
            return [getattr(eval(f"target{level_down}"), which)(maparg_, **kwargs)
                    for maparg_, target in
                        ( zip(mapargs, self.targets) if not show_progress else tqdm( zip(mapargs, self.targets) ) )
                        ]

        return [attr if not (callable(attr:=getattr(eval(f"target{level_down}"), which)) and allow_call) else\
                attr(**kwargs)
                for target in (self.targets if not show_progress else tqdm(self.targets)) ]

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def targets(self):
        """ list of individual targets """
        return self._targets

    @property
    def ntargets(self):
        """ number of targets in the sample """
        return len(self.targets)
