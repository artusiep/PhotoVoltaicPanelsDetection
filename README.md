# PhotoVoltaicPanelDetection

This repository contains solution for automatic detection of solar modules.

## Running in docker container

### Example run from docker

```
docker run -v ${PWD}/data:/usr/data pvpd:0.0.1 -c PlasmaConfig -o /usr/data/ -f /usr/data/plasma/1.JPG -t raw -cm plasma
```

- `-v ${PWD}/data:/usr/data` shares directory `data` from repository to `/usr/data`
- `pvpd:0.0.1` version of docker image
- `-c PlasmaConfig` which config will be used for image annotation
- `-o /usr/data/` data will be saved in the shared directory `data`
- `-f /usr/data/plasma/1.JPG` file `1.JPG` is going to be analyzing
- `-t raw` file `1.JPG` is in a raw format. There is a need to extract thermal information
- `-cm plasma` which color map will be used for thermal image values

### Build Docker Image from source

```
docker build -t pvpd:0.0.1 -t pvpd  .
```

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