from setuptools import setup, find_packages

setup(
    name="ai_cstq",
    version="0.1.0",
    description="BSGM-CellTrack: Bayesian Swin Graph Mamba end-to-end CTC cell tracker",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "tifffile>=2023.1.23",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "coco": ["pycocotools>=2.0.6"],
        "viz": ["matplotlib>=3.7.0"],
    },
)
