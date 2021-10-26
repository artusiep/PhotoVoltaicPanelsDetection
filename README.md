[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)
# PhotoVoltaicPanelDetection

This repository contains solution for automatic detection of solar modules. Main aim was 
to prepare a tool that can be configured and based on that configuration produce detected
modules in to be defined formats, commonly used in machine learning but not only. 

## Running in docker container

### Build Docker Image from source

```
docker build -t pvpd:0.0.1 -t pvpd  .
```

Try running:
```
docker run pvpd:0.1.0 -h 
```

### Example run from docker

```
docker run -v ${PWD}/data:/usr/data pvpd:0.1.0  -c PlasmaConfig -o /usr/data --f /usr/data/raw/3.JPG -t raw -cm plasma -l LabelMeLabeler 
```

- `-v ${PWD}/data:/usr/data` shares directory `data` from repository to `/usr/data`
- `pvpd:0.1.0` version of docker image
- `-c PlasmaConfig` which config will be used for image annotation
- `-o /usr/data/` data will be saved in the shared directory `data`
- `-f /usr/data/plasma/1.JPG` file `1.JPG` is going to be analyzing
- `-t raw` file `1.JPG` is in a raw format. There is a need to extract thermal information
- `-cm plasma` which color map will be used for thermal image values



## Local installation

### Prerequisites

- Required Python > 3.7. Check python version

```
python --version
```

- For extraction exif data (thermal data) from raw images: [exiftool](https://exiftool.org/)

### Installation

Install the needed python packages commands below:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### First run

```
./detector_cli.py
```

### Configs
Configs are parts of this project that every user adjust for private images.
All config can be found those in [config directory](detector/configs). Using them requires some
knowledge about classic image processing methods.

Whole detection is split into distinct steps that could be configured using each step 
`Param`. It is highly suggested taking a look at those. 

For now available Configs can be found using `./detector_cli.py -h`

### Labelers
Labeler is an entity that can be selected to generate a file containing data and metadata
from detection process. New labelers can be easily created to meet individual needs. All
labelers can be found in [labelers directory](detector/labelers)

For now available Labelers can be found using `./detector_cli.py -h` 
