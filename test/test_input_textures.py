import pytest
import numpy as np
from PyShaderPlayground.ShaderPlaygroundInputs import InputTextureSound

magnitude_test_data = [
    (1.0, 0.0), (1.0j, 0.0), (0.0, 0.0), (0.2+0.1j, 0.0),
    (-1.0, 0.0), (-1.0j, 0.0), (0.0, 0.0), (-0.2+0.1j, 0.0),
    (8.0+6.0j, 1.0), (8.0-6.0j, 1.0), (-8.0+6.0j, 1.0), (-8.0-6.0j, 1.0),
    (80.0+60.0j, 2.0), (80.0-60.0j, 2.0), (-80.0+60.0j, 2.0), (-80.0-60.0j, 2.0)
]
@pytest.mark.parametrize("x, expected", magnitude_test_data)
def test_magnitude_db(x, expected):
    assert InputTextureSound.calculate_magnitude_db(x) == expected
