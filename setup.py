"""
Setup configuration for TBFusionAI package.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_file = Path(__file__).parent / "readme.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="tbfusionai",
    version="1.0.0",
    author="AHFID AI Labs",
    author_email="aiteam@ahfid.org",
    description="AI-powered TB detection using cough sound analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahfidlabs/tbfusionai",
    packages=find_packages(exclude=["tests", "notebooks", "docker"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tbfusionai=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
