"""Setup for Agent"""

from setuptools import setup, find_packages

setup(
    name="social-arena-agent",
    version="1.0.0",
    author="Social-Arena",
    description="Minimal AI agent with 12 fundamental actions",
    url="https://github.com/Social-Arena/Agent",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["pydantic>=2.0"],
)
