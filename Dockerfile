FROM python:3.9.7-slim

WORKDIR /usr/app

RUN apt-get update && \
    apt-get -y install exiftool


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY detector_cli.py .
COPY detector/ detector

ENTRYPOINT ["python", "detector_cli.py"]