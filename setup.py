from setuptools import setup, find_packages

setup(
    name="autoTICI",
    version="0.0.1",
    author="Ruisheng Su",
    author_email="r.su@tue.nl",
    description="Automatic calculation of TICI score for DSA image",
    long_description=open("README.md").read(),
    url="https://github.com/RuishengSu/autoTICI",
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Modified MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)

