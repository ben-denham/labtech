FROM spark:4.2.0-python3
MAINTAINER ben@denham.nz

USER root

# Install uv, see:
# https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

# Set up home directory for spark user.
RUN mkdir /home/spark
RUN chown spark:spark /home/spark
RUN usermod -d /home/spark spark

USER spark

# Install python to match host .python-version because ray requires
# that even minor versions match.
RUN uv python install --default 3.11.5
COPY requirements.txt /home/spark/requirements.txt
RUN cd /home/spark && \
    uv venv && \
    uv pip install -r requirements.txt

# We could set the PYSPARK_PYTHON env variable instead, but that needs
# to be set from the driver (i.e. from jupyter if it's starting a full
# Spark driver), so it's simpler just to replace the system python.
USER root
RUN ln -sf /home/spark/.venv/bin/python3.11 /usr/local/bin/python3
USER spark
ENV PYTHONPATH=/home/spark/.venv/lib/python3.11/site-packages
ENV PATH="$PATH:/home/spark/.venv/bin"

# Allow spark user to access volume files by adding them to "shared"
# group with gid matching the host user.
USER root
RUN groupadd -g 1000 shared
RUN usermod -aG shared spark
USER spark
