"""Tests of the configuration reading."""
from slicersim.iotools import read_config, resolve_config_paths


def test_resolve_config_paths(tmp_path):
    """Data files named in a configuration are found next to it."""
    (tmp_path / "local.csv").write_text("wavelength,x\n")
    (tmp_path / "test.toml").write_text(
        '[spectrograph]\nthroughput = "local.csv"\ndispersion_law = "dispersion_default.ecsv"\n'
        'name = "test"\nlist = ["local.csv", "other.csv"]\n')
    config = read_config(str(tmp_path / "test.toml"))
    spec = config["spectrograph"]
    assert spec["throughput"] == str(tmp_path / "local.csv")               # local file
    assert spec["dispersion_law"] == "dispersion_default.ecsv"             # package file: unchanged
    assert spec["name"] == "test"                                          # not a file
    assert spec["list"] == [str(tmp_path / "local.csv"), "other.csv"]


def test_package_config_unchanged():
    """Nothing changes for the package configuration directory."""
    config = read_config("lazuli_cbe.toml")
    assert config["spectrograph"]["throughput"]["throughput"] == "throughput_bol.csv"
    assert resolve_config_paths({"a": "b"}, "/nonexistent") == {"a": "b"}
