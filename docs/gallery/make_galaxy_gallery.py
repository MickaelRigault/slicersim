"""Build the figures and tables of the galaxy-model page (docs/galaxy_models.rst).

The NGC galaxies come from the ``lazuli_astrosim_data`` package: an r-band
image (Legacy Surveys DR10 cutout, header from SGA-2020) and/or a spectral
template (Brown et al. 2014 atlas). Run from anywhere, after any change of
that package::

    python docs/gallery/make_galaxy_gallery.py

It writes, in ``docs/gallery/``:

- ``galaxies/<name>.png``: one gallery card per galaxy (image and/or spectrum);
- ``galaxies/table.inc`` and ``galaxies/cards_*.inc``: rst snippets included
  by ``galaxy_models.rst``;
- ``galaxy_redshifts.png``: a few galaxies at several redshifts in the
  Lazuli fields;
- ``galaxy_spectra.png``: all spectral templates.
"""
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import AsinhNorm
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmology

import lazuli_astrosim_data
from slicersim.scene.sources.galaxy import GalaxySpatialModel

DATA_PATH = pathlib.Path(lazuli_astrosim_data.__file__).parent.joinpath("data")
OUT_PATH = pathlib.Path(__file__).parent
CARD_PATH = OUT_PATH.joinpath("galaxies")

LAZULI_RANGE = (4_000, 17_000)  # observed wavelength range [Angstrom]
FIELDS = {"narrow": 0.04, "wide": 0.08}  # spaxel size [arcsec], 58 x 58 spaxels
NSPAXELS = 58
SPECTRUM_RANGE = (2_000, 25_000)  # rest-frame range shown on the cards [Angstrom]

CMAP = "cividis"


# ============== #
#  Data          #
# ============== #
def get_galaxy_names():
    """Galaxy names (``ngc0628`` format) with an image and/or a spectrum."""
    def _names(pattern, suffix):
        return {p.name.replace(suffix, "").replace("ngc_", "ngc")
                for p in DATA_PATH.joinpath("ngc").glob(pattern)}

    images = _names("ngc_*_thumbnails.fits.gz", "_thumbnails.fits.gz")
    spectra = _names("ngc_*_spec.fits", "_spec.fits")
    return sorted(images | spectra), images, spectra


def read_spectrum(name):
    """Rest-frame wavelength [Angstrom] and flux [erg/s/cm2/A] of the template."""
    data = fits.getdata(DATA_PATH.joinpath("ngc", f"{name.replace('ngc', 'ngc_')}_spec.fits"))
    return np.asarray(data["wavelength"], dtype=float), np.asarray(data["flux"], dtype=float)


def read_image(name):
    """r-band image [nanomaggy] and its header."""
    image, header = fits.getdata(DATA_PATH.joinpath("ngc", f"{name.replace('ngc', 'ngc_')}_thumbnails.fits.gz"),
                                 header=True)
    return np.clip(np.nan_to_num(image), 0, None), header


def get_catalog():
    """SGA-2020 entries of the NGC galaxies, indexed by ``ngc0628``-like names."""
    catalog = pd.read_parquet(DATA_PATH.joinpath("sga2020_catalog_of_ngcs.parquet"))
    catalog.index = catalog["Name"].str.lower().str.replace(" ", "")
    return catalog


def check_model(modelname, redshift=0.5):
    """Whether a slicersim host with this model gives a finite, non-zero cube."""
    from slicersim.scene.host import Host
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            host = Host.from_modelname(modelname, redshift=redshift)
            cube = host.get_cube(np.linspace(*LAZULI_RANGE, 50), shape=(NSPAXELS, NSPAXELS),
                                 pixel_scale=FIELDS["wide"])[1]
    except Exception:
        return False
    return bool(np.all(np.isfinite(cube)) and np.sum(cube) > 0)


def get_properties(names, images, spectra):
    """Table of galaxy properties used to choose a model."""
    catalog = get_catalog()
    rows = []
    for name in names:
        row = {"name": name, "image": name in images, "spectrum": name in spectra}
        sga = catalog.loc[name.upper().replace("_", "").lower()] if name.replace("_", "") in catalog.index else None
        if sga is not None:
            row |= {"sma_arcsec": sga["SMASB26"], "g_r": sga["gmagSB26"] - sga["rmagSB26"]}

        if name in images: # redshift used by slicersim to rescale the image
            _, header = read_image(name)
            row["redshift"] = header["REDSHIFT"]
        elif sga is not None:
            row["redshift"] = sga["z-LEDA"]

        if "sma_arcsec" in row and row.get("redshift", 0) > 0:
            # physical size, as slicersim sees it (angular diameter distance of the galaxy redshift)
            row["sma_kpc"] = row["sma_arcsec"] / cosmology.arcsec_per_kpc_proper(row["redshift"]).value

        if name in spectra: # rest-frame coverage of the template
            lbda, _ = read_spectrum(name)
            row["lbda_min"], row["lbda_max"] = lbda.min(), lbda.max()

        # does slicersim actually manage to use them?
        row["spectrum_ok"] = name in spectra and check_model(f"sersic:{name}")
        row["image_ok"] = name in images and check_model(f"{name}:bb-5000")

        rows.append(row)

    return pd.DataFrame(rows).set_index("name")


# ============== #
#  Figures       #
# ============== #
def _show_image(ax, name, size_arcsec=None):
    """r-band image, asinh stretch, with a 10 kpc scale bar."""
    image, header = read_image(name)
    vmax = np.percentile(image, 99.8)
    extent = np.array([-0.5, 0.5, -0.5, 0.5]) * image.shape[0] * header["PIXSCALE"] / 60 # [arcmin]
    ax.imshow(image, origin="lower", cmap=CMAP, extent=extent,
              norm=AsinhNorm(linear_width=vmax/50, vmin=0, vmax=vmax), interpolation="nearest")
    ax.set_facecolor(plt.get_cmap(CMAP)(0))
    ax.set_xticks([]); ax.set_yticks([])
    if header["REDSHIFT"] > 0:
        kpc10 = 10 * cosmology.arcsec_per_kpc_proper(header["REDSHIFT"]).value / 60 # [arcmin]
        x0, y0 = extent[0] * 0.9, extent[2] * 0.85
        ax.plot([x0, x0 + kpc10], [y0, y0], c="w", lw=1.5)
        ax.text(x0 + kpc10/2, y0 * 0.93, "10 kpc", color="w", ha="center", va="bottom", fontsize=7)


def _show_spectrum(ax, name, color="C0"):
    """Rest-frame template, normalized at 5500 A."""
    lbda, flux = read_spectrum(name)
    flag = (lbda > SPECTRUM_RANGE[0]) & (lbda < SPECTRUM_RANGE[1])
    norm = np.interp(5500, lbda, flux)
    ax.plot(lbda[flag] / 1e4, flux[flag] / norm, color=color, lw=0.8)
    ax.set_xlim(*np.asarray(SPECTRUM_RANGE) / 1e4)
    ax.set_yscale("log")
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xlabel("rest wavelength [µm]", fontsize=6, labelpad=1)
    ax.set_yticks([])
    ax.minorticks_off()


def make_card(name, properties):
    """Gallery card: image (if any) above the spectrum (if any)."""
    has_image, has_spectrum = properties.loc[name, ["image", "spectrum"]]
    fig = plt.figure(figsize=(2.2, 3.0 if has_image else 1.2), dpi=110)
    if has_image and has_spectrum:
        ax_img = fig.add_axes([0.02, 0.33, 0.96, 0.66])
        ax_spec = fig.add_axes([0.06, 0.1, 0.9, 0.2])
    elif has_image:
        ax_img, ax_spec = fig.add_axes([0.02, 0.02, 0.96, 0.96]), None
    else:
        ax_img, ax_spec = None, fig.add_axes([0.06, 0.28, 0.9, 0.68])

    if ax_img is not None:
        _show_image(ax_img, name)
    if ax_spec is not None:
        _show_spectrum(ax_spec, name, color="#1f4e79")

    fig.savefig(CARD_PATH.joinpath(f"{name}.png"))
    plt.close(fig)


def _color_of(g_r, vmin=0.3, vmax=1.05):
    """Color of a spectrum from the galaxy g-r color (blue to red)."""
    if not np.isfinite(g_r):
        return "0.4"
    return plt.get_cmap("coolwarm")(np.clip((g_r - vmin) / (vmax - vmin), 0, 1))


def make_spectra_figure(properties):
    """All templates, normalized at 5500 A, ordered by g-r color."""
    spectra = properties[properties["spectrum"]]
    # no g-r first (thin, grey), then from blue to red.
    names = list(spectra[spectra["g_r"].isna()].index) + list(spectra["g_r"].dropna().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        lbda, flux = read_spectrum(name)
        flag = (lbda > SPECTRUM_RANGE[0]) & (lbda < SPECTRUM_RANGE[1])
        no_color = not np.isfinite(properties.loc[name, "g_r"])
        ax.plot(lbda[flag] / 1e4, flux[flag] / np.interp(5500, lbda, flux),
                color="0.75" if no_color else _color_of(properties.loc[name, "g_r"]),
                lw=0.5 if no_color else 0.8, alpha=0.9)

    for z, ls in [(0, "-"), (1, "--")]:
        ax.axvspan(*np.asarray(LAZULI_RANGE) / 1e4 / (1 + z), color="0.85", alpha=0.5, zorder=-1, lw=0)
        ax.text(LAZULI_RANGE[0] / 1e4 / (1 + z), 0.012, f" Lazuli at z={z}", fontsize=8, color="0.4")

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.3, 1.05))
    fig.colorbar(sm, ax=ax, label="g − r (SGA-2020; grey: not in catalog)", pad=0.01)
    ax.set(xscale="log", yscale="log", xlim=np.asarray(SPECTRUM_RANGE) / 1e4, ylim=(1e-2, 20),
           xlabel="rest-frame wavelength [µm]", ylabel=r"$F_\lambda$ / $F_\lambda$(5500 Å)",
           title=f"The {len(names)} NGC spectral templates (Brown et al. 2014)")
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2], ["0.2", "0.3", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(OUT_PATH.joinpath("galaxy_spectra.png"), dpi=110)
    plt.close(fig)


def make_redshift_figure(names=("ngc5194", "ngc4486", "ngc4631", "ngc3310"),
                         redshifts=(0.1, 0.3, 0.6, 1.0), field="wide"):
    """A few galaxies as seen in a Lazuli field at several redshifts (no PSF)."""
    pixel_scale = FIELDS[field]
    fig, axes = plt.subplots(len(names), len(redshifts), figsize=(1.9 * len(redshifts) + 0.6, 1.9 * len(names)))
    for name, axes_ in zip(names, axes):
        model = GalaxySpatialModel.from_name(name)
        for z, ax in zip(redshifts, axes_):
            img = model.project_to(z, shape=(NSPAXELS, NSPAXELS), pixel_scale=pixel_scale)
            vmax = img.max()
            ax.imshow(img, origin="lower", cmap=CMAP, interpolation="nearest",
                      norm=AsinhNorm(linear_width=vmax/30, vmin=0, vmax=vmax))
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.97, f"{100 * img.sum():.0f}% of flux", transform=ax.transAxes,
                    color="w", fontsize=7, va="top")
            if name == names[0]:
                ax.set_title(f"z = {z}", fontsize=10)
        axes_[0].set_ylabel(name, fontsize=10)

    fig.suptitle(f"Lazuli {field} field ({NSPAXELS}×{NSPAXELS} spaxels of {pixel_scale}\") — galaxy image only, before PSF",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PATH.joinpath("galaxy_redshifts.png"), dpi=110)
    plt.close(fig)


# ============== #
#  rst snippets  #
# ============== #
def _fmt(value, fmt):
    return "–" if value is None or not np.isfinite(value) else format(value, fmt)


def get_modelname(row):
    """Recommended model name (None if neither the image nor the spectrum works)."""
    name = row.name
    if row["image_ok"] and row["spectrum_ok"]:
        return name
    if row["spectrum_ok"]:
        return f"sersic:{name}"
    if row["image_ok"]:
        return f"{name}:bb-5000"
    return None


def get_issue(row):
    """Short description of what does not work (None if all is fine)."""
    issues = []
    if row["image"] and not row["image_ok"]:
        issues.append("image (z ≤ 0)" if not row.get("redshift", 1) > 0 else "image")
    if row["spectrum"] and not row["spectrum_ok"]:
        issues.append("spectrum")
    return " and ".join(issues) + " unusable" if issues else None


def write_table(properties):
    """list-table of all galaxies."""
    lines = [".. list-table::",
             "   :header-rows: 1",
             "   :widths: 12 18 8 8 12 10 8 24",
             "",
             "   * - Name", "     - Model name", "     - Image", "     - Spectrum",
             "     - Own redshift", "     - SMA [kpc]", "     - g − r", "     - Note"]
    for name, row in properties.iterrows():
        modelname = get_modelname(row)
        lines += [f"   * - {name}",
                  f"     - ``{modelname}``" if modelname else "     - –",
                  f"     - {'yes' if row['image'] else '–'}",
                  f"     - {'yes' if row['spectrum'] else '–'}",
                  f"     - {_fmt(row.get('redshift'), '.4f')}",
                  f"     - {_fmt(row.get('sma_kpc'), '.0f')}",
                  f"     - {_fmt(row.get('g_r'), '.2f')}",
                  f"     - {get_issue(row) or ''}"]

    CARD_PATH.joinpath("table.inc").write_text("\n".join(lines) + "\n")


def write_cards(properties, which):
    """sphinx-design grid of gallery cards."""
    names = properties[properties["image"] & properties["spectrum"]].index if which == "full" else \
            properties[~properties["image"] & properties["spectrum"]].index
    columns = "2 2 3 3" if which == "full" else "2 3 3 4"
    lines = [f".. grid:: {columns}", "   :gutter: 2", ""]
    for name in names:
        row = properties.loc[name]
        modelname, issue = get_modelname(row), get_issue(row)
        lines += [f"   .. grid-item-card:: {name.upper()}",
                  "      :class-card: sd-shadow-sm" + (" sd-border-warning" if issue else ""),
                  "      :text-align: center",
                  "",
                  f"      .. image:: /gallery/galaxies/{name}.png",
                  f"         :alt: {name.upper()}" + (" r-band image and" if which == "full" else "") + " spectral template",
                  "",
                  f"      ``host=\"{modelname}\"``" if modelname else "      not usable yet",
                  "",
                  f"      SMA {_fmt(row.get('sma_kpc'), '.0f')} kpc · g−r {_fmt(row.get('g_r'), '.2f')}",
                  ""]
        if issue:
            lines += [f"      :bdg-warning:`{issue}`", ""]

    CARD_PATH.joinpath(f"cards_{which}.inc").write_text("\n".join(lines))


# ============== #
#  Main          #
# ============== #
if __name__ == "__main__":
    CARD_PATH.mkdir(exist_ok=True)
    names, images, spectra = get_galaxy_names()
    properties = get_properties(names, images, spectra)

    for name in names:
        make_card(name, properties)

    make_spectra_figure(properties)
    make_redshift_figure()
    write_table(properties)
    write_cards(properties, "full")
    write_cards(properties, "spectrum")

    print(f"{len(names)} galaxies ({len(images)} images, {len(spectra)} spectra) -> {OUT_PATH}")
    print(properties.describe().T[["min", "50%", "max"]])
    problems = properties.apply(get_issue, axis=1).dropna()
    print(f"{len(problems)} galaxies with an unusable part:")
    print(problems.to_string())
