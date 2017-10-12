import pytest
import numpy as np


def test_declination():
    """ Test the declination ufunc """
    from pypet import declination

    # Create a python equivalent of the declination function
    def py_declination(day_of_year):
        return np.pi*(23.45/180.0) * np.sin(2*np.pi*(284 + day_of_year)/365)
    py_declination = np.vectorize(py_declination)

    # Test with a single value ...
    np.testing.assert_allclose(declination(1), py_declination(1))

    # ... and array (int64) ...
    doy = np.arange(366)
    np.testing.assert_allclose(declination(doy), py_declination(doy))

    # ... and array (int32)
    doy = doy.astype(np.int32)
    np.testing.assert_allclose(declination(doy), py_declination(doy))


def test_sunset_hour_angle():
    """ sunset_hour_angle ufunc """
    from pypet import sunset_hour_angle

    def py_sunset_hour_angle(latitude, declination):
        return np.acos(-np.tan(latitude)*np.tan(declination))

    np.testing.assert_allclose(sunset_hour_angle(0.0, 0.0), sunset_hour_angle(0.0, 0.0))
    np.testing.assert_allclose(sunset_hour_angle(np.pi/4, np.pi/4), sunset_hour_angle(np.pi/4, np.pi/4))


def test_daily_extraterrestrial_radiation():
    from pypet import daily_extraterrestrial_radiation, declination, sunset_hour_angle

    def py_daily_extraterrestrial_radiation(day_of_year, latitude):
        """
        Compute daily total extraterrestrial radiation in MJ m-2 day-1

        Reference,
            Duffie & Beckman, Solar Engineering of Thermal Processes (Fourth Edition), 2013
                Equation 1.10.3
        """
        solar_constant = 1367  # W/m2

        dec = declination(day_of_year)
        ssha = sunset_hour_angle(latitude, dec)

        H = 24 * 3600 * solar_constant / np.pi
        H *= 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        H *= np.cos(latitude) * np.cos(dec) * np.sin(ssha) + ssha * np.sin(latitude) * np.sin(dec)
        return H / 1e6  # Convert to MJ m-2 day-1
    py_daily_extraterrestrial_radiation = np.vectorize(py_daily_extraterrestrial_radiation)

    np.testing.assert_allclose(daily_extraterrestrial_radiation(1, 0.0), py_daily_extraterrestrial_radiation(1, 0.0))


