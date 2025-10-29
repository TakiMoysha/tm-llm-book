# ===========================================================
# config
# ===========================================================
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MinioObjectStorageConfig:
    url: str = field(default_factory=lambda: os.getenv("S3_API_URL"))
    access_key: str = field(default_factory=lambda: os.getenv("S3_ACCESS_KEY"))
    secret_key: str = field(default_factory=lambda: os.getenv("S3_SECRET_KEY"))
    secure: bool = field(default_factory=lambda: os.getenv("S3_SECURE") == "true")
    bucket_name: str = "devlab-llm-rfc_proto"
    object_prefix: str = ""


# ===========================================================
# exceptions
# ===========================================================
class StorageError(Exception): ...


class FileNotFoundError(StorageError): ...


# ===========================================================
# protocols
# ===========================================================

from abc import ABC, abstractmethod
from typing import AsyncContextManager, BinaryIO


class ObjectStorage(ABC):
    @abstractmethod
    async def upload(self, key: str, data: BinaryIO, *, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    async def download(self, key: str) -> AsyncContextManager[BinaryIO]: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...


# ===========================================================
# Minio Client
# ===========================================================

import aioboto3

from botocore.exceptions import ClientError
from typing import AsyncContextManager, AsyncIterator
from contextlib import asynccontextmanager

from minio import Minio


class MinioObjectStorage(ObjectStorage):
    def __init__(self, config: MinioObjectStorageConfig) -> None:
        self.config = config
        self.session = aioboto3.Session()

    @property
    def client(self):
        return self.session.client(
            "s3",
            endpoint_url=self.config.url,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            use_ssl=self.config.secure,
        )

    def _add_prefix(self, key: str) -> str:
        prefix = self.config.object_prefix.strip("/")
        return f"{prefix}/{key}".strip("/") if prefix else key

    async def upload(self, key: str, data: BinaryIO, *, content_type: str = "application/octet-stream") -> None:
        key = self._add_prefix(key)
        async with self.client as client:
            try:
                await client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                )
            except ClientError as e:
                raise StorageError(f"Upload failed: {e}") from e

    @asynccontextmanager
    async def download(self, key: str) -> AsyncIterator[BinaryIO]:
        key = self._add_prefix(key)
        async with self.client as client:
            try:
                response = await client.get_object(Bucket=self.config.bucket_name, Key=key)
                yield response["Body"]
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"File not found: {key}")
                raise StorageError(f"Download failed: {e}") from e

    async def delete(self, key: str) -> None:
        key = self._add_prefix(key)
        async with self.client as client:
            try:
                await client.delete_object(Bucket=self.config.bucket_name, Key=key)
            except ClientError as e:
                raise StorageError(f"Delete failed: {e}") from e

    async def exists(self, key: str) -> bool:
        key = self._add_prefix(key)
        async with self.client as client:
            try:
                await client.head_object(Bucket=self.config.bucket_name, Key=key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    return False
                raise StorageError(f"Exists failed: {e}") from e


# ===========================================================
# examples
# ===========================================================
from msgspec import Struct  # or pydantic.BaseModel


class ObjectDTO(Struct):
    key: str
    url: str


class FileStorageService:
    def __init__(self, storage: ObjectStorage) -> None:
        self.storage = storage

    async def save(self, key: str, file: BinaryIO, content_type: str) -> None:
        await self.storage.upload(key, file, content_type=content_type)

    async def get(self, key: str) -> AsyncContextManager[BinaryIO]:
        return await self.storage.download(key)

    async def remove_file(self, key: str) -> None:
        await self.storage.delete(key)


def object_storage_client_service(config: MinioObjectStorageConfig) -> FileStorageService:
    return FileStorageService(MinioObjectStorage(config))


# ===========================================================
# tests
# ===========================================================

import pytest
from unittest.mock import AsyncMock

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def minio_client():
    config = MinioObjectStorageConfig()
    minio_storage = MinioObjectStorage(config=config)
    async with minio_storage.client as s3:
        try:
            await s3.head_bucket(Bucket=config.bucket_name)
        except ClientError:
            pytest.fail(f"Bucket {config.bucket_name} does not exist. Create it first.")
        yield minio_storage


# @pytest.mark.integration
# async def test_storage(storage: FileStorageService):
#     async with storage.get("test.txt") as file:
#         assert await file.read() == b"Hello, world!"


# @pytest.mark.integration
# async def test_upload():
#     mock_storage = AsyncMock(spec=ObjectStorage)
#     service = FileStorageService(mock_storage)
#
#     await service.save("test.txt", AsyncMock(), "text/plain")
#
#     mock_storage.upload.assert_called_once_with("test.txt", AsyncMock(), content_type="text/plain")
#     mock_storage.upload.assert_awaited()


# @pytest.mark.integration
# async def test_minio_connection_and_upload(minio_client: MinioObjectStorage):
#     key = "test/hello.txt"
#     data = b"Hello, MinIO!"
#
#     try:
#         await minio_client.upload(key, AsyncMock(), content_type="text/plain")
#     except Exception as e:
#         pytest.fail(f"Upload failed: {e}")
#
#     assert await minio_client.exists(key) is True
#
#     try:
#         async with minio_client.download(key) as stream:
#             downloaded = await stream.read()
#             assert downloaded == data
#     except Exception as e:
#         pytest.fail(f"Download failed: {e}")
#
#     await minio_client.delete(key)
#     assert await minio_client.exists(key) is False


@pytest.mark.integration
async def test_minio_nonexistent_file(minio_client: MinioObjectStorage):
    key = "not/exist.txt"

    print(minio_client)

    assert await minio_client.exists(key) is False

    with pytest.raises(FileNotFoundError):
        async with minio_client.download(key):
            pass
