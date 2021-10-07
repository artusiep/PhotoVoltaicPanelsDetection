# PhotoVoltaicPanelDetection

This repository contains solution for automatic detection of solar modules.

## Prerequisites

- Required Python > 3.7
- For extraction exif data (thermal data) from raw images: [exiftool](https://exiftool.org/)

## Local installation
Check python version
```
python --version
```
if python version is above 3.7 then:

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