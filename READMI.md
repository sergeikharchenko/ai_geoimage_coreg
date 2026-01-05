# GeoBind

AI-powered georeferencing for historical satellite imagery (Hexagon) using LoFTR and SuperPoint+SuperGlue.

## Installation

1. **Install GDAL**: Ensure GDAL is installed on your system.
   - Ubuntu: `sudo apt install libgdal-dev`
   - Conda: `conda install gdal`

2. **Clone SuperGlue** (Required if using SuperGlue algorithm):
   ```bash
   git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
   export PYTHONPATH=$PYTHONPATH:$(pwd)/SuperGluePretrainedNetwork