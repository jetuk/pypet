import pytest
import numpy as np


def test_oudin():
    from pypet import oudin, daily_extraterrestrial_radiation

    doy = 1
    latitude = 0.0

    def py_oudin(d, l, t, density=1000, latent_heat=2.45):
        re = daily_extraterrestrial_radiation(d, l)
        if t > -5:
            return 1e3 * re * (t + 5) / (100 * latent_heat * density)
        return 0.0
    py_oudin = np.vectorize(py_oudin)

    T = 10
    np.testing.assert_allclose(oudin(doy, latitude, T), py_oudin(doy, latitude, T))

    T = np.linspace(-10, 10)
    np.testing.assert_allclose(oudin(doy, latitude, T), py_oudin(doy, latitude, T))



