FROM spark:4.2.0-python3
MAINTAINER ben@denham.nz

USER root

RUN apt update && apt install -y \
    python3.11 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN python3.11 -m pip install -r requirements.txt

USER spark

ENV PYSPARK_PYTHON=/usr/bin/python3.11
