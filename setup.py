
from setuptools import setup, find_packages

requirements = [

]

setup(
    name='experiment-tools',
    version='0.1',
    packages=find_packages(exclude=['my_test']),
    include_package_data=True,
    install_requires=requirements,
    test_suite='my_test',
)
