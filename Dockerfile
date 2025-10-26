FROM python:3.12.6-slim

WORKDIR /usr/app

RUN apt-get update && \
    apt-get -y install exiftool


COPY requirements.txt .
RUN pip install -r requirements.txt

ENV TF_CPP_MIN_LOG_LEVEL=3

COPY .dockerignore .
COPY detector_cli.py .
COPY detector/ detector

ENTRYPOINT ["python", "detector_cli.py"]