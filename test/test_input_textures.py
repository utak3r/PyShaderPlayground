import pytest
import numpy as np
from PyShaderPlayground.ShaderPlaygroundInputs import InputTextureSound

def test_calculate_spectrum_shape():
    # Test that calculate_spectrum returns 512 bins for a 2048 sample input
    signal = np.random.rand(2048)
    spectrum = InputTextureSound.calculate_spectrum(signal)
    assert len(spectrum) == 512
    assert np.all(spectrum >= 0.0)

def test_get_audio_part_padding():
    # Test that get_audio_part pads with zeros if at the end
    audio = np.ones(100)
    part = InputTextureSound.get_audio_part(audio, time_start=0.0, sample_rate=100, num_samples=200)
    assert len(part) == 200
    assert np.all(part[:100] == 1.0)
    assert np.all(part[100:] == 0.0)
