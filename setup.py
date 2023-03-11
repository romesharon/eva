from setuptools import setup, find_packages

setup(
    name='eva_',
    version='0.0.1',
    description='An evaluation package',
    author='Rom Sharon and Meni Shakarov',
    license='BSD 2-clause',
    packages=['eva'] + ['eva.' + pkg for pkg in find_packages('eva')],
    install_requires=['cycler==0.11.0',
                    'joblib==1.1.1',
                    'kiwisolver==1.3.1',
                    'matplotlib==3.5.0',
                    'numpy==1.21.0',
                    'Pillow==8.4.0',
                    'pyparsing==2.4.7',
                    'python-dateutil==2.8.2',
                    'scikit-learn==1.2.1',
                    'scipy==1.6.0',
                    'six==1.16.0',
                    'threadpoolctl==3.1.0',
                    "jedi>=0.16"
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
