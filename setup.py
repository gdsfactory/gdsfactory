from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f.readlines() if not line.strip().startswith("-")
    ]

with open("requirements_dev.txt") as f:
    requirements_dev = [
        line.strip() for line in f.readlines() if not line.strip().startswith("-")
    ]

with open("requirements_full.txt") as f:
    requirements_full = [
        line.strip() for line in f.readlines() if not line.strip().startswith("-")
    ]

with open("requirements_exp.txt") as f:
    requirements_exp = [
        line.strip() for line in f.readlines() if not line.strip().startswith("-")
    ]

with open("README.md") as f:
    long_description = f.read()


setup(
    name="gdsfactory",
    url="https://github.com/gdsfactory/gdsfactory",
    version="5.9.0",
    author="gdsfactory community",
    scripts=["gdsfactory/gf.py"],
    description="python library to generate GDS layouts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.7",
    license="MIT",
    entry_points="""
        [console_scripts]
        gf=gdsfactory.gf:gf
    """,
    extras_require={
        "full": list(set(requirements + requirements_full)),
        "basic": requirements,
        "dev": list(set(requirements + requirements_dev + requirements_full)),
        "exp": list(set(requirements + requirements_exp)),
    },
    package_data={
        "": ["*.gds", "*.yml", "*.lyp", "*.json"],
    },
)
