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

with open("requirements_sipann.txt") as f:
    requirements_sipann = [
        line.strip() for line in f.readlines() if not line.strip().startswith("-")
    ]

with open("requirements_tidy3d.txt") as f:
    requirements_tidy3d = [
        line.strip() for line in f.readlines() if not line.strip().startswith("-")
    ]

with open("README.md") as f:
    long_description = f.read()


setup(
    name="gdsfactory",
    url="https://github.com/gdsfactory/gdsfactory",
    version="5.39.0",
    author="gdsfactory community",
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
        "sipann": requirements_sipann,
        "tidy3d": requirements_tidy3d,
        "dev": list(set(requirements + requirements_dev)),
        "exp": list(set(requirements + requirements_exp)),
    },
    package_data={
        "": ["*.gds", "*.yml", "*.lyp", "*.json"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Framework :: Pytest",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
)
