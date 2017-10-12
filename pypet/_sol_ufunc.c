#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "_sol_func.c"

/*
 * These are ufuncs for computing solar extraterrestrial radiation.
 *
 * The source of these equations is:
 *      Duffie & Beckman, Solar Engineering of Thermal Processes (Fourth Edition), 2013
 *           Equation 1.6.1a
 */

static PyMethodDef SolMethods[] = {
        {NULL, NULL, 0, NULL}
};

/*
 *   Declination, the angular position of the sun at solar noon (i.e., when the sun is on the
 *   local meridian) with respect to the plane of the equator, north positive; −23.45 deg ≤ δ ≤ 23.45 deg
 *
 */

static void int32_declination(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    npy_int32 day_of_year;

    for (i = 0; i < n; i++) {
        day_of_year = *(npy_int64 *)in;
        *((double *)out) = _int64_declination(day_of_year);
        in += in_step;
        out += out_step;
    }
}

static void int64_declination(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    npy_int64 day_of_year;

    for (i = 0; i < n; i++) {
        day_of_year = *(npy_int64 *)in;
        *((double *)out) = _int64_declination(day_of_year);
        in += in_step;
        out += out_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction declination_funcs[2] = {
    &int32_declination,
    &int64_declination
};

static char declination_types[4] = {
    NPY_INT32, NPY_DOUBLE,
    NPY_INT64, NPY_DOUBLE
};

/*
 *   Sunset hour angle
 *
 */


static void double_sunset_hour_angle(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    double latitude, declination;

    for (i = 0; i < n; i++) {
        latitude = *(double *)in1;
        declination = *(double *)in2;
        *((double *)out1) = _double_sunset_hour_angle(latitude, declination);

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

static void float_sunset_hour_angle(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    float latitude, declination;

    for (i = 0; i < n; i++) {
        latitude = *(float *)in1;
        declination = *(float *)in2;
        *((float *)out1) = _float_sunset_hour_angle(latitude, declination);

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction sunset_hour_angle_funcs[2] = {
    &double_sunset_hour_angle,
    &float_sunset_hour_angle
};

static char sunset_hour_angle_types[6] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
};


/*
 *
 *   Compute daily total extraterrestrial radiation in MJ m-2 day-1
 *
 *       Duffie & Beckman, Solar Engineering of Thermal Processes (Fourth Edition), 2013
 *           Equation 1.10.3
 *
 */


static void double_daily_extraterrestrial_radiation(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    double latitude;
    npy_int64 day_of_year;

    for (i = 0; i < n; i++) {
        day_of_year = *(npy_int64 *)in1;
        latitude = *(double *)in2;

        *((double *)out1) = _double_daily_extraterrestrial_radiation(day_of_year, latitude);

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction radiation_funcs[1] = {
    &double_daily_extraterrestrial_radiation,
};

static char radiation_types[3] = {
    NPY_INT64, NPY_DOUBLE, NPY_DOUBLE,
};


static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_sol_ufunc",
    NULL,
    -1,
    SolMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__sol_ufunc(void)
{
    PyObject *m, *declination, *sunset_hour_angle, *daily_extraterrestrial_radiation, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    declination = PyUFunc_FromFuncAndData(declination_funcs, data, declination_types, 2, 1, 1,
                                    PyUFunc_None, "declination", "declination_docstring", 0);
    sunset_hour_angle = PyUFunc_FromFuncAndData(sunset_hour_angle_funcs, data, sunset_hour_angle_types, 2, 2, 1,
                                    PyUFunc_None, "sunset_hour_angle", "sunset_hour_angle_docstring", 0);
    daily_extraterrestrial_radiation = PyUFunc_FromFuncAndData(radiation_funcs, data, radiation_types, 1, 2, 1,
                                    PyUFunc_None, "daily_extraterrestrial_radiation", "daily_extraterrestrial_radiation_docstring", 0);
    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "declination", declination);
    PyDict_SetItemString(d, "sunset_hour_angle", sunset_hour_angle);
    PyDict_SetItemString(d, "daily_extraterrestrial_radiation", daily_extraterrestrial_radiation);

    Py_DECREF(declination);
    Py_DECREF(sunset_hour_angle);
    Py_DECREF(daily_extraterrestrial_radiation);

    return m;
}
#else
PyMODINIT_FUNC init_sol_ufunc(void)
{
    PyObject *m, *declination, *sunset_hour_angle, *daily_extraterrestrial_radiation, *d;


    m = Py_InitModule("_sol_ufunc", SolMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    declination = PyUFunc_FromFuncAndData(declination_funcs, data, declination_types, 2, 1, 1,
                                    PyUFunc_None, "declination", "declination_docstring", 0);
    sunset_hour_angle = PyUFunc_FromFuncAndData(sunset_hour_angle_funcs, data, sunset_hour_angle_types, 2, 2, 1,
                                    PyUFunc_None, "sunset_hour_angle", "sunset_hour_angle_docstring", 0);
    daily_extraterrestrial_radiation = PyUFunc_FromFuncAndData(radiation_funcs, data, radiation_types, 1, 2, 1,
                                    PyUFunc_None, "daily_extraterrestrial_radiation", "daily_extraterrestrial_radiation_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "declination", declination);
    PyDict_SetItemString(d, "sunset_hour_angle", sunset_hour_angle);
    PyDict_SetItemString(d, "daily_extraterrestrial_radiation", daily_extraterrestrial_radiation);

    Py_DECREF(declination);
    Py_DECREF(sunset_hour_angle);
    Py_DECREF(daily_extraterrestrial_radiation);
}
#endif