from setuptools import setup, find_packages

setup(
    name="rabitq_abv",
    version="0.1.0",
    description="Adaptive Bit-Width Vector Quantization (ABV-Quant) and RaBitQ Benchmarking Suite",
    author="Adaptive VQ Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "tabulate>=0.8.9",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
    ],
)
