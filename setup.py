from setuptools import setup, find_packages
ver = {}
exec(open('delta_lfit_2/version.py').read(), None, ver)

setup(
    name = 'delta_lfit_2',
    packages = find_packages(exclude=[]),
    include_package_data = True,
    version = ver['__version__'],
    author = 'Yin Jun Phua',
    author_email = 'phua@c.titech.ac.jp',
    install_requires=[
        'torch>=2.1,<2.2',
        'numpy>=1.26,<1.27',
        'scipy>=1.11,<1.12',
        'click>=8.1,<8.2',
        'tqdm>=4.66,<4.67',
        'h5py>=3.10,<3.11',
        'redis>=5.0,<5.1',
    ],
)
