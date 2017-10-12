import numpy


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('pypet', parent_package, top_path)
    config.add_extension('_sol_ufunc', ['pypet/_sol_ufunc.c'])
    config.add_extension('_pet_ufunc', ['pypet/_pet_ufunc.c'])

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name='pypet',
        version='0.1dev',
        description='Python Potential Evapotranspiration functions.',
        author='James E Tomlinson',
        author_email='tomo.bbe@gmail.com',
        packages=['pypet', ],
        configuration=configuration
    )
