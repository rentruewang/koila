from pathlib import Path

import setuptools

WorkDir = Path(__file__).parent

with open(WorkDir / "README.md") as f:
    long_desc = f.read()

with open(WorkDir / "requirements.txt") as f:
    requires = f.readlines()

setuptools.setup(
    name="koila",
    version="0.1",
    author="RenChu Wang",
    description="Prevent PyTorch's `CUDA error: out of memory` in one line of code.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["examples", "tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    py_modules=["koila"],
    install_requires=requires,
)
