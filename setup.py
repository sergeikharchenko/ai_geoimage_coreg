from setuptools import setup, find_packages

setup(
    name="ai_geoimage_coreg",
    version="0.1.0",
    description="Automated Georeferencing using LoFTR and SuperPoint+SuperGlue",
    author="Sergei Kharchenko",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "numpy",
        "opencv-python",
        "rasterio",
        "pandas",
        "geopandas",
        "matplotlib",
        "kornia",
        "shapely",
        # Note: GDAL is required but often hard to install via pip alone.
        # It is usually assumed to be installed via conda or system packages.
    ],
    entry_points={
        'console_scripts': [
            'ai_geoimage_coreg=ai_geoimage_coreg.core:main_cli',
        ],
    },
)
