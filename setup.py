"""
setup.py for Whisper Voice — setuptools-based packaging.
"""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

# Read requirements
requirements = []
req_file = here / "requirements.txt"
if req_file.exists():
    requirements = [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

# Read long description from README if present
long_description = ""
readme = here / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")

setup(
    name="whisper-voice",
    version="0.1.0",
    author="whisper-voice",
    description="Real-time speech-to-text via OpenAI Whisper API with global hotkey",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "whisper-voice = src.app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        "src": ["*.json"],
    },
)
