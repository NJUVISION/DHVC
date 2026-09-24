import platform
import sys
from pathlib import Path

from setuptools import Distribution, find_packages, setup


if {"bdist_wheel", "editable_wheel", "install", "develop"}.intersection(sys.argv):
    if (sys.platform != "linux" or platform.machine().lower() != "x86_64"
            or sys.version_info[:2] != (3, 12)):
        raise RuntimeError(
            "The bundled DHIC extension requires Linux x86_64 and CPython 3.12"
        )


class BinaryDistribution(Distribution):
    """Mark the bundled CPython extension as platform and ABI specific."""

    def has_ext_modules(self):
        return True


setup(
    name="dhic",
    version="1.0.0",
    description="Hierarchical neural image codec",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license_files=["LICENSE.txt"],
    packages=find_packages(),
    package_data={"dhic._native": ["*.so"]},
    install_requires=["torch", "numpy", "Pillow", "compressai", "scipy"],
    python_requires=">=3.12,<3.13",
    distclass=BinaryDistribution,
)
