from setuptools import setup, find_packages

setup(
    name="datascienceproject",
    version="0.1",
    packages=find_packages(include=['sections', 'sections.*']),
    package_dir={'sections': 'sections'},
    install_requires=[
        "streamlit==1.24.0",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "seaborn==0.11.2",
        "pillow==9.0.1",
        "matplotlib==3.5.1",
        "unidecode==1.3.6"
    ],
    python_requires='>=3.9',
) 