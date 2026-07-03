import numpy as np
import pandas
from astropy import units as u

def broadcast_mapping(value, ntargets):
    """Broadcast a value to a given number of targets.

    Parameters
    ----------
    value : array-like
        The value(s) to broadcast.
    ntargets : int
        The number of targets to broadcast to.

    Returns
    -------
    ndarray
        The broadcasted values with shape (ntargets,) or (ntargets, value.shape[-1]).
    """
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
    """Build an interpolator for exposure time as a function of redshift.

    Creates a smoothing spline that interpolates exposure time values for Type Ia
    supernovae across a range of redshifts, given a target signal-to-noise ratio.

    Parameters
    ----------
    snr : float
        Target signal-to-noise ratio.
    per_resolution : bool, optional
        If True, compute SNR per resolution element. Default is True.
    redshift_range : list of float, optional
        The redshift range [z_min, z_max] for the reference sample.
        Default is [0.01, 1.5].
    ntrails : int, optional
        Number of redshift points to sample in the redshift range. Default is 20.
    **kwargs
        Additional keyword arguments passed to setup_to_snr.

    Returns
    -------
    scipy.interpolate.UnivariateSpline
        A smoothing spline that interpolates exposure time [in seconds] given redshift.
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
    """A collection of targets for simulation with a slicer spectrograph.

    This class manages multiple targets and provides methods to setup spectrograph
    configurations, retrieve exposure time and data volume estimates, and execute
    operations across all targets in the sample.
    """

    def __init__(self, targets, setup_spectrograph=True):
        """Initialize a Sample with a list of targets.

        Parameters
        ----------
        targets : list
            List of target objects to include in the sample.
        setup_spectrograph : bool, optional
            If True, automatically setup the spectrograph for each target.
            Default is True.
        """
        self._targets = targets
        if setup_spectrograph:
            self.setup_spectrograph()

    @classmethod
    def from_sneia(cls, redshifts, colors=0, stretchs=0, **kwargs):
        """Create a Sample from Type Ia supernova parameters.

        Parameters
        ----------
        redshifts : array-like
            Redshift values for the supernovae.
        colors : float or array-like, optional
            Color parameter(s) for the supernovae. Default is 0.
        stretchs : float or array-like, optional
            Stretch parameter(s) for the supernovae. Default is 0.
        **kwargs
            Additional keyword arguments passed to LazuliSupernova initialization.

        Returns
        -------
        Sample
            A new Sample instance containing LazuliSupernova targets.
        """
        from . import LazuliSupernova
        data = pandas.DataFrame({"redshift": redshifts,
                                 "stretch": stretchs,
                                 "color": colors})
        targets = [LazuliSupernova(**(this.to_dict() | kwargs)) for _, this in data.iterrows()]
        return cls(targets)

    # ========== #
    #  methods   #
    # ========== #
    # -------- #
    #  GETTER  #
    # -------- #
    def get_target(self, index):
        """Retrieve a target from the sample by index.

        Parameters
        ----------
        index : int
            The index of the target to retrieve.

        Returns
        -------
        object
            The target at the specified index.
        """
        return self.targets[index]

    def get_exposure_time(self, units="second", full=False, **kwargs):
        """Get exposure times for all targets in the sample.

        Parameters
        ----------
        units : str, optional
            The units for the exposure time. Default is "second".
        full : bool, optional
            If True, return full exposure time information. Default is False.
        **kwargs
            Additional keyword arguments passed to target.get_exposure_time.

        Returns
        -------
        ndarray
            Exposure times for all targets in the requested units.
        """
        return np.asarray( self.call_down("get_exposure_time", full=full, **kwargs)) * u.second.to(units)

    def get_data_volume(self, units="GB"):
        """Get data volume estimates for all targets in the sample.

        Parameters
        ----------
        units : str, optional
            The units for the data volume. Default is "GB".

        Returns
        -------
        ndarray
            Data volumes for all targets in the requested units.
        """
        return np.asarray( self.call_down("get_data_volume") )

    def get_readout_config(self):
        """Get readout configuration for all targets in the sample.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing readout configuration parameters for all targets,
            including 'ngroups', 'nframes_per_group', and 'ndrops'.
        """
        readout = pandas.DataFrame(self.call_down("get_readout_config"))
        nmd = readout.pop("nmd")
        return readout.join(pandas.DataFrame(np.stack(nmd.values),
                            columns=["ngroups", "nframes_per_group", "ndrops"])
                           )

    def get_summary_stats(self):
        """Get summary statistics for all targets in the sample.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing data volume, exposure time, and readout configuration
            for all targets in the sample.
        """
        data_volume = self.get_data_volume()
        exposure_time = self.get_exposure_time("s")
        readout_config = self.get_readout_config()

        return pandas.DataFrame({"volume":data_volume,
                                  "exptime":exposure_time,
                                 }).join(readout_config)

    def get_total_surveyduration(self, sample_size=8000, fraction_survey=0.4, units="yr"):
        """Estimate the total survey duration for observing a large sample.

        Parameters
        ----------
        sample_size : int, optional
            The total number of targets in the survey. Default is 8000.
        fraction_survey : float, optional
            The fraction of the survey time available for observations. Default is 0.4.
        units : str, optional
            The units for the survey duration. Default is "yr".

        Returns
        -------
        float
            The estimated total survey duration.
        """
        return self.get_exposure_time(units).sum() * sample_size/self.ntargets / fraction_survey

    # -------- #
    #  SETUP   #
    # -------- #
    def change_property(self, **kwargs):
        """Change simulation properties for all targets in the sample.

        Parameters
        ----------
        **kwargs
            Properties and values to update for each target's simulation.

        Returns
        -------
        list
            Results from updating each target's simulation.
        """
        return [target.simulation.update(**kwargs) for target in self.targets]


    # -------- #
    #  SETUP   #
    # -------- #
    def setup_spectrograph(self, redshift_cut=0.8):
        """Setup the spectrograph mode for each target based on redshift.

        Targets with redshift <= redshift_cut are configured with 'narrow' mode,
        while targets with higher redshifts use 'wide' mode.

        Parameters
        ----------
        redshift_cut : float, optional
            The redshift threshold for switching between narrow and wide modes.
            Default is 0.8.
        """
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
        """Setup all targets to achieve a target signal-to-noise ratio.

        Parameters
        ----------
        snr : float
            Target signal-to-noise ratio.
        per_resolution : bool, optional
            If True, compute SNR per resolution element. Default is True.
        lbda_range : list of float, optional
            The wavelength range [lambda_min, lambda_max] in Angstroms.
            Default is [4000, 6800].
        frame : str, optional
            Reference frame for wavelengths, either 'rest' or 'observed'.
            Default is 'rest'.
        statistic : callable, optional
            Function to compute the summary statistic for SNR. Default is np.mean.
        inplace : bool, optional
            If True, modify target configurations in place. Default is True.
        client : optional
            A dask client for distributed computation. If None, uses serial computation.
            Default is None.
        show_progress : bool, optional
            If True, display a progress bar. Default is False.
        **kwargs
            Additional keyword arguments passed to target.setup_to_snr.

        Returns
        -------
        tuple
            A tuple of (configs_dataframe, snrs_array) where configs_dataframe is a
            pandas.DataFrame of detector configurations and snrs_array is an ndarray
            of achieved SNR values.
        """


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
        """Call a method on each target in the sample.

        Parameters
        ----------
        which : str
            The name of the method to call on each target.
        mapargs : array-like, optional
            Arguments to map over targets. If provided, each target receives
            a corresponding argument from mapargs. Default is None.
        allow_call : bool, optional
            If True, call methods; if False, return method objects. Default is True.
        level_down : str, optional
            Specifies a nested attribute level to apply the method call to,
            e.g., "simulation" to call on target.simulation. Default is "".
        show_progress : bool, optional
            If True, display a progress bar during execution. Default is False.
        **kwargs
            Additional keyword arguments passed to the called method.

        Returns
        -------
        list
            Results from calling the method on each target.
        """
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
        """list of individual targets"""
        return self._targets

    @property
    def ntargets(self):
        """number of targets in the sample"""
        return len(self.targets)
