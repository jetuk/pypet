#include "math.h"


static double _int32_declination(npy_int32 day_of_year) {
    return M_PI*(23.45/180.0) * sin(2*M_PI*(284 + day_of_year)/365);
}

static double _int64_declination(npy_int64 day_of_year) {
    return M_PI*(23.45/180.0) * sin(2*M_PI*(284 + day_of_year)/365);
}

static double _double_sunset_hour_angle(double latitude, double declination) {
    return acos(-tan(latitude)*tan(declination));
}

/* Solar constant in W/m2 */
double SOLAR_CONSTANT_DOUBLE = 1367;
float SOLAR_CONSTANT_FLOAT = 1367;

static double _double_daily_extraterrestrial_radiation(npy_int64 day_of_year, double latitude) {
    double H, dec, ssha;
    H = 24 * 3600 * SOLAR_CONSTANT_DOUBLE / M_PI;
    H *= 1 + 0.033*cos(2*M_PI*day_of_year/365);

    dec = _int64_declination(day_of_year);
    ssha = _double_sunset_hour_angle(latitude, dec);

    H *= cos(latitude)*cos(dec)*sin(ssha) + ssha*sin(latitude)*sin(dec);
    return H / 1e6;  // Convert to MJ m-2 day-1
}