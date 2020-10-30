from setuptools import find_packages, setup


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if not line.startswith("-")]


setup(
    name="gdsmp",
    url="https://github.com/gdsfactory/gdsfactory",
    version="1.4.4",
    author="PsiQ",
    description="simulate GDS in meep",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_install_requires(),
    python_requires=">=3.6",
)
