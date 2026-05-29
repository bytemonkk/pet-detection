from setuptools import setup

setup(
    name="pet-detection-cnn",
    version="1.0.0",
    author="Manoj Kumar Sunkara",
    description="Cat vs Dog Image Classification using Convolutional Neural Networks",
    install_requires=[
        "tensorflow",
        "keras",
        "keras-preprocessing",
        "numpy",
        "matplotlib",
        "pillow",
        "opencv-python",
        "scikit-learn"
    ],
    python_requires=">=3.10",
)