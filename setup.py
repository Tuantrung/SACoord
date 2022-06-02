from setuptools import setup

requirements = [
    'PyYAML',
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'networkx',
    'tqdm',
    'Click'
]

setup(
    name='sacoord',
    version='0.0.1',
    packages=['sacoord'],
    url='https://github.com/Tuantrung/SACoord',
    license='None',
    author='Tuantrung',
    python_requires=">=3.6.*, <3.8.*",
    author_email='tuantrung03121997@gmail.com',
    install_requires=requirements,
    description='This is my implement using simulated annealing algorithm to solve coord problem',
    entry_points={
        'console_scripts': [
            "sacoord=sacoord.main:cli",
        ],
    },
)
