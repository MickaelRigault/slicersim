from slicersim import thermal

def test_ThermalOptics():
    """Test the ThermalOptics class."""
    optics = thermal.ThermalOptics(temperature=300, emissivity=0.1, fratio=10)
    assert optics is not None


def test_ThermalRadiation():
    """Test the ThermalRadiation class."""
    radiation = thermal.ThermalRadiation(temperature=300, emissivity=0.1)
    assert radiation is not None
