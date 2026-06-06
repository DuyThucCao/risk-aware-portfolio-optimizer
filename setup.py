"""Package metadata for the Risk-Aware Portfolio Optimizer."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent


def read_text(filename: str) -> str:
    """Read a project text file with a safe fallback for packaging tools."""
    path = PROJECT_ROOT / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_requirements(filename: str = "requirements.txt") -> list[str]:
    """Parse runtime requirements while ignoring comments and blank lines."""
    requirements: list[str] = []
    for line in read_text(filename).splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        requirements.append(requirement)
    return requirements


setup(
    name="risk-aware-portfolio-optimizer",
    version="1.0.0",
    author="Thuc Cao",
    description=(
        "Risk-aware portfolio optimization with transaction costs, CVaR, "
        "volatility targeting, and walk-forward backtesting."
    ),
    long_description=read_text("README.md"),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/yourusername/risk-aware-portfolio-optimizer",
    project_urls={
        "Source": "https://github.com/yourusername/risk-aware-portfolio-optimizer",
        "Documentation": "https://github.com/yourusername/risk-aware-portfolio-optimizer#readme",
    },
    packages=find_packages(include=["src", "src.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "portfolio-optimization",
        "risk-management",
        "backtesting",
        "cvar",
        "volatility-targeting",
        "transaction-costs",
        "quant-finance",
    ],
    entry_points={
        "console_scripts": [
            "rao-backtest=src.backtesting.backtester:main",
        ],
    },
)
