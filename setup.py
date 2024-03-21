import setuptools
import subprocess
import os

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md"), "r") as fh:
    long_description = fh.read()


pytorch_sidu_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

if "-" in pytorch_sidu_version:
    # when not on tag, git describe outputs: "1.3.3-22-gdf81228"
    # pip has gotten strict with version numbers
    # so change it to: "1.3.3+22.git.gdf81228"
    # See: https://peps.python.org/pep-0440/#local-version-segments
    v,i,s = pytorch_sidu_version.split("-")
    pytorch_sidu_version = v + "+" + i + ".git." + s

assert "-" not in pytorch_sidu_version
assert "." in pytorch_sidu_version

assert os.path.isfile("cf_remote/version.py")
with open("cf_remote/VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % pytorch_sidu_version)

setuptools.setup(
    name="pytorch-sidu",
    version=pytorch_sidu_version,
    author="Marco Parola",
    author_email="marcoparola96@gmail.com",
    description="SIDU: SImilarity Difference and Uniqueness method for explainable AI",
    url="https://github.com/MarcoParola/pytorch-sidu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["pytorch-sidu"],
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
)
