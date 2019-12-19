from setuptools import setup


setup(  name="machine_learning",
        version="0.0.1",
        author="Zsolt Forray",
        license="MIT",
        packages="machine_learning",
        install_requires=[
            "numpy == 1.16.3",
            "scikit-learn == 0.19.1",
            "scipy == 0.19.1",
            ],
    )
