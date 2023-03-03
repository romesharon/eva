from setuptools import setup

setup(
    name='eva_rom_meni',
    version='0.1.0',
    description='An evaluation package',
    author='Rom Sharon and Meni Shakarov',
    license='BSD 2-clause',
    packages=['src'],
    install_requires=['cycler==0.11.0',
                    'joblib==1.1.1',
                    'kiwisolver==1.3.1',
                    'matplotlib==3.3.4',
                    'numpy==1.19.5',
                    'Pillow==8.4.0',
                    'pyparsing==2.4.7',
                    'python-dateutil==2.8.2',
                    'scikit-learn==0.24.2',
                    'scipy==1.5.4',
                    'six==1.16.0',
                    'threadpoolctl==3.1.0',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)