#!/usr/bin/env python3
"""
Setup script for Computer Vision Trading Agents.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="computer-vision-trading-agents",
    version="1.0.0",
    author="Tyler Leake",
    author_email="jtylerleake10@gmail.com",
    description="A sophisticated algorithmic trading system combining computer vision and machine learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jtylerleake/computer-vision-trading-agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cv-trading=main:main",
            "cv-trading-quick=examples.quick_start:quick_start_example",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "trading",
        "algorithmic-trading",
        "machine-learning",
        "reinforcement-learning",
        "computer-vision",
        "gaf",
        "financial-analysis",
        "quantitative-finance",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/computer-vision-trading-agents/issues",
        "Source": "https://github.com/yourusername/computer-vision-trading-agents",
        "Documentation": "https://github.com/yourusername/computer-vision-trading-agents/wiki",
    },
)
