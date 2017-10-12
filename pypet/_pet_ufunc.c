#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "_sol_func.c"

/*
 * These are ufuncs for computing potential evapo-transpiration using a variety of methods.
 *
 */

static PyMethodDef PetMethods[] = {
        {NULL, NULL, 0, NULL}
};

/*
 *  Oudin (2005) formulation.
 *
 */


// density of water (kg m-3)
double WATER_DENSITY_DOUBLE = 1000;


// the latent heat flux (MJ kg−1)
double WATER_LATENT_HEAT_FLUX_DOUBLE = 2.45;


static double _double_oudin(npy_int64 day_of_year, double latitude, double air_temperature) {
    double H, pet;

    // Compute extra-terrestrial radiation
    H = _double_daily_extraterrestrial_radiation(day_of_year, latitude);  // MJ m-2 day-1

    // Now compute PET
    if (air_temperature > -5.0) {
        pet = H / (WATER_LATENT_HEAT_FLUX_DOUBLE * WATER_DENSITY_DOUBLE);
        pet *= (air_temperature + 5.0) / 100.0;
        pet *= 1000.0;  // m/day -> mm/day
    } else {
        pet = 0.0;
    }
    return pet;
}

static void double_oudin(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *in3 = args[2];
    char *out1 = args[3];
    npy_intp in1_step = steps[0], in2_step = steps[1], in3_step = steps[2];
    npy_intp out1_step = steps[3];

    double latitude, air_temperature;
    npy_int64 day_of_year;

    for (i = 0; i < n; i++) {
        day_of_year = *(npy_int64 *)in1;
        latitude = *(double *)in2;
        air_temperature = *(double *)in3;

        *((double *)out1) = _double_oudin(day_of_year, latitude, air_temperature);

        in1 += in1_step;
        in2 += in2_step;
        in3 += in3_step;
        out1 += out1_step;
    }
}



/*This a pointer to the above function*/
PyUFuncGenericFunction oudin_funcs[1] = {
    &double_oudin
};

static char oudin_types[4] = {
    NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE
};



static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_pet_ufunc",
    NULL,
    -1,
    PetMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__pet_ufunc(void)
{
    PyObject *m, *oudin, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    oudin = PyUFunc_FromFuncAndData(oudin_funcs, data, oudin_types, 1, 3, 1,
                                    PyUFunc_None, "oudin", "oudin_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "oudin", oudin);

    Py_DECREF(oudin);

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
    sunset_hour_angle = PyUFunc_FromFuncAndData(declination_funcs, data, declination_types, 2, 2, 1,
                                    PyUFunc_None, "sunset_hour_angle", "sunset_hour_angle_docstring", 0);
    daily_extraterrestrial_radiation = PyUFunc_FromFuncAndData(radiation_funcs, data, radiation_types, 1, 2, 1,
                                    PyUFunc_None, "daily_extraterrestrial_radiation", "daily_extraterrestrial_radiation_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "declination", declination);
    PyDict_SetItemString(d, "sunset_hour_angle", sunset_hour_angle);
    PyDict_SetItemString(d, "daily_extraterrestrial_radiation", daily_extraterrestrial_radiation);

    Py_DECREF(declination);
}
#endif