import setuptools
import os

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md"), "r") as fh:
    long_description = fh.read()

# read requirements file
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pytorch-sidu",
    version="0.0.0",
    author="Marco Parola",
    author_email="marcoparola96@gmail.com",
    description="SIDU: SImilarity Difference and Uniqueness method for explainable AI",
    url="https://github.com/MarcoParola/pytorch-sidu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["pytorch_sidu", "pytorch_sidu.utils"],
    install_requires=requirements,
    setuptools_git_versioning={
        "enabled": True,
    },
    setup_requires=[
        "setuptools-git-versioning<2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
)
