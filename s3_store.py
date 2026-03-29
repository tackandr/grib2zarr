"""s3_store.py - Helpers for opening local or S3-backed Zarr stores.

When an output path starts with ``s3://`` the helpers here create an
:class:`zarr.storage.FsspecStore` backed by *s3fs*.  S3 credentials are read
from the following environment variables:

* ``AWS_ACCESS_KEY_ID``     – S3 / AWS access key ID
* ``AWS_SECRET_ACCESS_KEY`` – S3 / AWS secret access key
* ``S3_HOSTNAME``           – Custom endpoint URL (optional; for non-AWS S3
  implementations such as MinIO).  When absent, the default AWS endpoints are
  used.
"""
from __future__ import annotations

import os
import shutil
from typing import Union

import s3fs
import zarr


def _get_s3_storage_options() -> dict:
    """Build fsspec storage options from environment variables.

    Returns
    -------
    dict
        Options suitable for passing as *storage_options* to
        :meth:`zarr.storage.FsspecStore.from_url`.
    """
    options: dict = {}
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("S3_HOSTNAME")
    if key:
        options["key"] = key
    if secret:
        options["secret"] = secret
    if endpoint_url:
        options["endpoint_url"] = endpoint_url
    return options


def open_store(path: str) -> Union[str, zarr.storage.FsspecStore]:
    """Return a zarr-compatible store for *path*.

    For local paths the path string is returned unchanged (zarr handles it
    natively).  For ``s3://`` URIs a :class:`zarr.storage.FsspecStore` is
    returned, configured with credentials sourced from the environment
    variables ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and
    optionally ``S3_HOSTNAME`` (custom S3 endpoint URL).

    Parameters
    ----------
    path:
        Local filesystem path or ``s3://bucket/key`` URI.

    Returns
    -------
    str or zarr.storage.FsspecStore
    """
    if not path.startswith("s3://"):
        return path
    return zarr.storage.FsspecStore.from_url(
        path, storage_options=_get_s3_storage_options()
    )


def delete_store(path: str) -> None:
    """Delete a local directory or an S3 path recursively.

    For local paths :func:`shutil.rmtree` is used.  For ``s3://`` URIs the
    *s3fs* library removes all objects under the prefix.

    Parameters
    ----------
    path:
        Local filesystem path or ``s3://bucket/key`` URI.
    """
    if path.startswith("s3://"):
        options = _get_s3_storage_options()
        fs = s3fs.S3FileSystem(
            key=options.get("key"),
            secret=options.get("secret"),
            endpoint_url=options.get("endpoint_url"),
        )
        if fs.exists(path):
            fs.rm(path, recursive=True)
    else:
        shutil.rmtree(path, ignore_errors=True)
