FROM python:3.12-slim

# Install the eccodes C library required by the eccodes Python binding
RUN apt-get update \
    && apt-get install -y --no-install-recommends libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the project files and install the package
COPY pyproject.toml ./
COPY grib2zarr.py config_parser.py ./

RUN pip install --no-cache-dir .

ENTRYPOINT ["grib2zarr"]
