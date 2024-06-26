#!/usr/bin/env python

import slicersim
import os
import numpy as np
import matplotlib.pyplot as plt
from twins_embedding import TwinsEmbeddingModel
from manifold_gp import ManifoldGaussianProcess
from iminuit import Minuit, cost
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table
import pickle

# initialize model for fitting
model = TwinsEmbeddingModel()


def trained_model(wave, dm, av, xi1, xi2, xi3, phase):
    flux, flux_error = model.evaluate(magnitude=dm, color=av,
                                      coordinates=[xi1, xi2, xi3], phase=phase)
    flux_interp = np.interp(wave, model.wave, flux)
    return flux_interp


def plot_results(X):
    redshift = X[:, 0]
    params = X[:, 1:7]
    params_fit = X[:, 7:]
    labels = [
        '$\\Delta m$',
        '$c$',
        '$\\xi_1$',
        '$\\xi_2$',
        '$\\xi_3$',
        '$\\Delta t$',
    ]

    # add in derived delta(distance modulus) using the RBTL GP manifold
    if os.path.exists('rbtl_gp.pkl'):
        with open('rbtl_gp.pkl', 'rb') as f:
            manifold = pickle.load(f)
    elif os.path.exists('apjabec3ct3_mrt.txt'):  # from https://content.cld.iop.org/journals/0004-637X/912/1/70/revision2/apjabec3ct3_mrt.txt
        datas = Table.read('apjabec3ct3_mrt.txt', format='ascii.cds')
        mask_color = datas['DAv'] < 0.5
        covariates = np.array([datas['DAv']])
        manifold = ManifoldGaussianProcess(None, np.array(datas[['xi1', 'xi2', 'xi3']].as_array().tolist()),
                                           datas['Dm'], datas['e_Dm'], covariates=covariates,
                                           mask=mask_color, parameters=None)
        manifold.fit()
    else:
        manifold = None

    if manifold is not None:
        dmu_true = manifold.predict(params[:, 2:5], prediction_covariates=params[:, 1], return_uncertainties=False)
        dmu_fit = manifold.predict(params_fit[:, 2:5], prediction_covariates=params_fit[:, 1], return_uncertainties=False)
        params = np.hstack([params, dmu_true[:, None]])
        params_fit = np.hstack([params_fit, dmu_fit[:, None]])
        labels.append('$\\Delta \\mu$')

    varied = params.std(axis=0).astype(bool)
    params = params[:, varied]
    diffs = params_fit[:, varied] - params
    labels = np.array(labels)[varied]
    nextra = 2  # histogram & redshift

    fig, axes = plt.subplots(params.shape[1] + nextra, diffs.shape[1], figsize=(9., 10.), sharex='col', sharey='row')

    for diff, axcol, collabel in zip(diffs.T, axes.T, labels):
        axcol[0].hist(diff, bins='auto')
        axcol[0].set_title(f'${diff.mean():.2f} \\pm {diff.std():.2f}$')
        s = axcol[1].scatter(diff, redshift, c=redshift, marker='.')
        for p, ax, rowlabel in zip(params.T, axcol[nextra:], labels):
            ax.scatter(diff, p, c=redshift, marker='.')

    axes[0, 0].set_ylabel('$N$')
    axes[1, 0].set_ylabel('$z$')
    for i, label in enumerate(labels):
        axes[i + nextra, 0].set_ylabel(label)
        axes[-1, i].set_xlabel(f'Error in {label}')

    fig.tight_layout()
    cax = fig.colorbar(s, ax=axes, fraction=0.1, aspect=30.)
    cax.set_label('Redshift $z$')
    return fig


if __name__ == '__main__':
    # initialize model for simulation
    config = slicersim.iotools.get_config(scene='twins_embedding.toml')
    sim = slicersim.Simulation.from_config(config)

    # generate random twins embedding parameters
    N = 100
    rng = np.random.default_rng()
    z_test = 2. * rng.power(3., size=(N, 1))  # z=0 to z=2 with z**2 distribution
    dm_test = rng.normal(0., 0.2, size=(N, 1))
    color_test = rng.normal(0., 0.2, size=(N, 1))
    xi_test = rng.multivariate_normal(mean=[0., 0., 0.], cov=np.diag([2., 2., 2.]), size=N)
    phase_test = np.zeros((N, 1))  # fix to 0 for now
    grid = np.hstack([z_test, dm_test, color_test, xi_test, phase_test])

    outpath = 'te_grid'
    os.makedirs(outpath, exist_ok=True)

    results = []
    for redshift, magnitude, color, xi1, xi2, xi3, phase in grid:
        # simulate spectrum with new parameters
        sim.scene.update(
            target__redshift=redshift,
            target__magnitude=magnitude,
            target__color=color,
            target__coordinates=[xi1, xi2, xi3],
            target__phase=phase,
        )
        wl_in, flux_in, var_in = sim.get_spectrum(incl_error=True)

        # remove NaNs
        finite = np.isfinite(flux_in)
        wl_in = wl_in[finite]
        flux_in = flux_in[finite]
        var_in = var_in[finite]

        # calibrate the spectrum and scale to z=0.05, shift the wavelength to the rest frame
        sensfunc = np.interp(wl_in, *sim.get_effective_transmission()) * 1e-15
        background = sim.scene.background.get_spectrum(wl_in)[1] / 1e-15
        dist_ratio = cosmo.luminosity_distance(redshift) / cosmo.luminosity_distance(0.05)
        cosmo_k_corr = (1. + redshift) / 1.05

        xfit = wl_in / (1. + redshift)
        yfit = flux_in * dist_ratio ** 2. * cosmo_k_corr / sensfunc - background
        wfit = np.sqrt(var_in) * dist_ratio ** 2. * cosmo_k_corr / sensfunc

        # set up and run the fit
        m = Minuit(cost.LeastSquares(xfit, yfit, wfit, trained_model),

                   # initial guesses
                   dm=0.,
                   av=0.,
                   xi1=0.,
                   xi2=0.,
                   xi3=0.,
                   phase=0.,

                   # fix parameters?
                   fix_dm=False,
                   fix_av=False,
                   fix_xi1=False,
                   fix_xi2=False,
                   fix_xi3=False,
                   fix_phase=True,

                   # initial step sizes
                   error_dm=0.05,
                   error_av=0.03,
                   error_xi1=0.5,
                   error_xi2=0.2,
                   error_xi3=0.5,

                   # parameter limits
                   limit_dm=(-10., 10.),
                   limit_av=(-0.4, 0.7),
                   limit_xi1=(-10., 10.),
                   limit_xi2=(-10., 10.),
                   limit_xi3=(-10., 10.),
                   )

        m.migrad()
        results.append(m.args)

        # plot the results
        fig, ax = plt.subplots()

        param_spec = '''$\\Delta m = {magnitude:.2f}$, $A_V = {color:.2f}$,
        $\\xi = [{coordinates[0]:.2f}, {coordinates[1]:.2f}, {coordinates[2]:.2f}]$, $\\Delta t = {phase:.2f}$'''

        ax.plot(xfit, yfit, label='data', lw=1)
        ax.fill_between(xfit, yfit - wfit, yfit + wfit, alpha=0.3)

        best_fit = trained_model(model.wave, *m.args)
        ax.plot(model.wave, best_fit, label='best fit: ' + param_spec.format(magnitude=m.args[0], color=m.args[1],
                                                                             coordinates=m.args[2:5], phase=m.args[5]))

        right_answer = trained_model(model.wave,
                                     sim.scene.target.meta['magnitude'],
                                     sim.scene.target.meta['color'],
                                     *sim.scene.target.meta['coordinates'],
                                     sim.scene.target.meta['phase']
                                     )
        ax.plot(model.wave, right_answer, label='right answer: ' + param_spec.format(**sim.scene.target.meta))

        ax.set_xlabel('Rest Wavelength (Å)')
        ax.set_ylabel('$F_λ$ at $z=0.05$ ($10^{-15}$ erg/s/cm²/Å)')
        ax.set_title(f'Twins Embedding Simulation at $z={redshift:.3f}$')
        ax.legend()

        fig.tight_layout()
        filename = f'{outpath}/te_z{redshift:.3f}_m{magnitude:.3f}_c{color:.3f}_xa{xi1:.3f}_xb{xi2:.3f}_xc{xi3:.3f}_t{phase:.3f}'.replace('.', 'p').replace('-', 'n') + '.pdf'
        fig.savefig(filename)
        plt.close(fig)
        print('saved', filename)

    X = np.hstack([grid, results])
    np.savetxt(f'{outpath}/results.txt', X)

    fig = plot_results(X)
    fig.savefig(f'{outpath}/results.pdf')
    plt.close(fig)
