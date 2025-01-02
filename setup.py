from setuptools import setup, find_packages

setup(
    name="adaptive-preformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "mne>=0.24.0",
        "numpy>=1.19.2",
        "tqdm>=4.62.3",
        "pyyaml>=5.4.1",
    ],
    author="Xingqiang Chen",
    author_email="your.email@example.com",
    description="Adaptive Preprocessing Transformer for EEG Signal Processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chenxingqiang/adaptive-preformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)