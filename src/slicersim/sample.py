"""Samples of targets, to estimate exposure times and data volumes in bulk.

`Sample` gathers several targets (typically
`~slicersim.lazuli.LazuliSupernova`) and calls their methods on all of them,
e.g. to set them all up to a given SNR and summarise the resulting exposure
times, read-out modes and data volumes. `build_exptime_estimator` uses it to
build an exposure time versus redshift relation for SNe Ia.
"""
import numpy as np
import pandas
from astropy import units as u


def broadcast_mapping(value, ntargets):
    """Broadcast a value to a given number of targets.

    Parameters
    ----------
    value : float or array_like
        The value(s) to broadcast.
    ntargets : int
        The number of targets to broadcast to.

    Returns
    -------
    numpy.ndarray
        The broadcast (read-only) values, of shape ``(ntargets,)`` if
        ``value`` is a scalar or 1D, ``(ntargets, value.shape[-1])`` if it
        is multi-dimensional.

    Examples
    --------
    >>> broadcast_mapping(0.5, 3)
    array([0.5, 0.5, 0.5])
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
    """Build an interpolator of the SN Ia exposure time as a function of redshift.

    A reference `Sample` of SNe Ia (``x1=0``, ``c=0``, see
    `Sample.from_sneia`) evenly spread over ``redshift_range`` is set up to
    reach ``snr``, and a smoothing spline is fitted to the resulting exposure
    times.

    Parameters
    ----------
    snr : float
        Target signal-to-noise ratio.
    per_resolution : bool, optional
        If True, ``snr`` is given per spectral resolution element, otherwise
        per wavelength bin. Default is True.
    redshift_range : list of float, optional
        Redshift range ``[z_min, z_max]`` of the reference sample.
        Default is ``[0.01, 1.5]``.
    ntrails : int, optional
        Number of redshifts sampled in ``redshift_range``. Default is 20.
    **kwargs
        Goes to `Sample.setup_to_snr` (e.g. ``lbda_range``, ``frame``).

    Returns
    -------
    scipy.interpolate.BSpline
        Smoothing spline giving the exposure time in seconds as a function of
        redshift.

    Examples
    --------
    >>> estimator = build_exptime_estimator(10, ntrails=5)  # doctest: +SKIP
    >>> exptime = estimator(0.8)  # in seconds  # doctest: +SKIP
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

class Sample:
    """A collection of targets handled together.

    This class manages multiple targets and provides methods to set up their
    spectrograph field and read-out configuration, retrieve exposure time and
    data volume estimates, and call any target method on all of them (see
    `call_down`).

    Parameters
    ----------
    targets : list
        Targets of the sample, e.g. `~slicersim.lazuli.LazuliSupernova`
        instances.
    setup_spectrograph : bool, optional
        If True, set the Lazuli field of each target according to its
        redshift (see `setup_spectrograph`). This requires Lazuli targets
        with a ``redshift``. Default is True.

    Examples
    --------
    >>> sample = Sample.from_sneia([0.3, 0.6, 1.0])  # doctest: +SKIP
    >>> configs, snrs = sample.setup_to_snr(10)  # doctest: +SKIP
    >>> stats = sample.get_summary_stats()  # doctest: +SKIP
    """

    def __init__(self, targets, setup_spectrograph=True):
        """Initialize a Sample with a list of targets.

        Parameters
        ----------
        targets : list
            Targets of the sample.
        setup_spectrograph : bool, optional
            If True, set the Lazuli field of each target according to its
            redshift (see `setup_spectrograph`). Default is True.
        """
        self._targets = targets
        if setup_spectrograph:
            self.setup_spectrograph()

    @classmethod
    def from_sneia(cls, redshifts, colors=0, stretchs=0, **kwargs):
        """Create a Sample of SALT Type Ia supernovae.

        Parameters
        ----------
        redshifts : array_like
            Redshifts of the supernovae.
        colors : float or array_like, optional
            SALT color ``c`` of the supernovae. Default is 0.
        stretchs : float or array_like, optional
            SALT stretch ``x1`` of the supernovae. Default is 0.
        **kwargs
            Goes to `~slicersim.lazuli.LazuliSupernova`, for all targets
            (e.g. ``phase``).

        Returns
        -------
        Sample
            A sample of `~slicersim.lazuli.LazuliSupernova` targets, with
            their field set up according to redshift (see
            `setup_spectrograph`).
        """
        from . import LazuliSupernova
        data = pandas.DataFrame({"redshift": redshifts,
                                 "x1": stretchs,
                                 "c": colors})
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
            The target at ``index``.
        """
        return self.targets[index]

    def get_exposure_time(self, units="second", full=False, **kwargs):
        """Get the total exposure time of each target.

        Parameters
        ----------
        units : str, optional
            Unit of the returned times (any astropy time unit, e.g. ``"s"``,
            ``"hr"``). Default is ``"second"``.
        full : bool, optional
            Passed to `~slicersim.target.VirtualTarget.get_exposure_time`.
            Only False is currently supported: the dicts returned with True
            cannot be converted to ``units``. Default is False.
        **kwargs
            Goes to `~slicersim.target.VirtualTarget.get_exposure_time`.

        Returns
        -------
        numpy.ndarray
            Exposure time of each target, in ``units``.
        """
        return np.asarray( self.call_down("get_exposure_time", full=full, **kwargs)) * u.second.to(units)

    def get_data_volume(self, units="GB"):
        """Get the data volume of each target.

        Parameters
        ----------
        units : str, optional
            Unit of the data volume. Currently ignored: volumes are always
            returned in GB. Default is ``"GB"``.

        Returns
        -------
        numpy.ndarray
            Data volume of each target in GB, all ramps included.
        """
        return np.asarray( self.call_down("get_data_volume") )

    def get_readout_config(self):
        """Get the read-out configuration of each target.

        Returns
        -------
        pandas.DataFrame
            One row per target, with columns ``"nramps"``, ``"ngroups"``,
            ``"nframes_per_group"`` and ``"ndrops"``.
        """
        readout = pandas.DataFrame(self.call_down("get_readout_config"))
        nmd = readout.pop("nmd")
        return readout.join(pandas.DataFrame(np.stack(nmd.values),
                            columns=["ngroups", "nframes_per_group", "ndrops"])
                           )

    def get_summary_stats(self):
        """Get the data volume, exposure time and read-out mode of each target.

        Returns
        -------
        pandas.DataFrame
            One row per target, with columns ``"volume"`` (GB), ``"exptime"``
            (s), ``"nramps"``, ``"ngroups"``, ``"nframes_per_group"`` and
            ``"ndrops"``.
        """
        data_volume = self.get_data_volume()
        exposure_time = self.get_exposure_time("s")
        readout_config = self.get_readout_config()

        return pandas.DataFrame({"volume":data_volume,
                                  "exptime":exposure_time,
                                 }).join(readout_config)

    def get_total_surveyduration(self, sample_size=8000, fraction_survey=0.4, units="yr"):
        """Estimate the duration of a survey observing ``sample_size`` targets.

        The mean exposure time of the sample is scaled to ``sample_size``
        targets and divided by the fraction of the survey time spent on these
        observations: ``sum(exptime) * sample_size / ntargets / fraction_survey``.

        Parameters
        ----------
        sample_size : int, optional
            Total number of targets in the survey. Default is 8000.
        fraction_survey : float, optional
            Fraction of the survey time spent observing these targets.
            Default is 0.4.
        units : str, optional
            Unit of the returned duration (any astropy time unit).
            Default is ``"yr"``.

        Returns
        -------
        float
            Survey duration in ``units``.
        """
        return self.get_exposure_time(units).sum() * sample_size/self.ntargets / fraction_survey

    # -------- #
    #  SETUP   #
    # -------- #
    def change_property(self, **kwargs):
        """Change simulation parameters of all targets.

        Parameters
        ----------
        **kwargs
            Parameters to update, as ``name=value``, passed to
            `~slicersim.simulation.Simulation.update` for each target.

        Returns
        -------
        list
            Output of `~slicersim.simulation.Simulation.update` for each
            target (None).
        """
        return [target.simulation.update(**kwargs) for target in self.targets]


    # -------- #
    #  SETUP   #
    # -------- #
    def setup_spectrograph(self, redshift_cut=0.8):
        """Set the Lazuli field of each target according to its redshift.

        Targets with ``redshift <= redshift_cut`` are observed in the
        ``"narrow"`` field, the others in the ``"wide"`` field (see
        `~slicersim.lazuli.VirtualLazuliTarget.change_spectrograph`).

        Parameters
        ----------
        redshift_cut : float, optional
            Redshift threshold between the narrow and wide fields.
            Default is 0.8.

        Returns
        -------
        None
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
        """Find, for each target, the read-out configuration reaching a target SNR.

        Parameters
        ----------
        snr : float
            Target signal-to-noise ratio.
        per_resolution : bool, optional
            If True, ``snr`` is given per spectral resolution element,
            otherwise per wavelength bin. Default is True.
        lbda_range : list of float, optional
            Wavelength range ``[min, max]`` in Angstrom over which the SNR is
            computed. Default is ``[4000, 6800]``.
        frame : str, optional
            Reference frame of ``lbda_range``: ``"rest"`` or ``"obs"``.
            Default is ``"rest"``.
        statistic : callable, optional
            Function used to summarise the SNR over ``lbda_range``.
            Default is `numpy.mean`.
        inplace : bool, optional
            If True, each target is updated to its configuration.
            Default is True.
        client : ``dask.distributed.Client``, optional
            Dask client to distribute the computation over targets. If None,
            targets are processed serially. Default is None.
        show_progress : bool, optional
            If True, display a tqdm progress bar (serial mode only).
            Default is False.
        **kwargs
            Goes to `~slicersim.target.VirtualTarget.setup_to_snr`.

        Returns
        -------
        configs : pandas.DataFrame
            Read-out configuration of each target, with columns ``"nmd"``
            and ``"nramps"``.
        snrs : numpy.ndarray
            SNR reached for each target.
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
        """Call a method (or get an attribute) of each target.

        Parameters
        ----------
        which : str
            Name of the method (or attribute) of each target.
        mapargs : array_like, optional
            Values mapped over targets: each target's method receives the
            corresponding entry as first positional argument (see
            `broadcast_mapping`). Default is None.
        allow_call : bool, optional
            If True, callable attributes are called; if False, they are
            returned as is. Ignored if ``mapargs`` is given.
            Default is True.
        level_down : str, optional
            Nested attribute of each target to apply ``which`` to, e.g.
            ``"simulation"`` to use ``target.simulation``. Default is ``""``.
        show_progress : bool, optional
            If True, display a tqdm progress bar. Default is False.
        **kwargs
            Goes to the called method.

        Returns
        -------
        list
            Output for each target.

        Examples
        --------
        >>> redshifts = sample.call_down("get_properties", "redshift")  # doctest: +SKIP
        >>> lbdas = sample.call_down("lbda", level_down="simulation.spectrograph")  # doctest: +SKIP
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
        """Targets of the sample (list)."""
        return self._targets

    @property
    def ntargets(self):
        """Number of targets in the sample (int)."""
        return len(self.targets)
