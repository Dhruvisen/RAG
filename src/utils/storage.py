import boto3
import logging
from botocore.exceptions import ClientError
from typing import Optional

logger = logging.getLogger(__name__)

MINIO_ENDPOINT = "http://localhost:9010"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "rag-documents"

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name="us-east-1",
        )
        # Create bucket if it doesn't exist
        try:
            _s3_client.head_bucket(Bucket=BUCKET_NAME)
        except ClientError:
            try:
                _s3_client.create_bucket(Bucket=BUCKET_NAME)
                logger.info("Created MinIO bucket: %s", BUCKET_NAME)
            except Exception as e:
                logger.error("Could not create MinIO bucket: %s", e)
    return _s3_client

def upload_to_minio(object_name: str, data: bytes) -> bool:
    s3 = get_s3_client()
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=object_name, Body=data)
        return True
    except ClientError as e:
        logger.error("MinIO upload failed: %s", e)
        return False

def get_from_minio(object_name: str) -> Optional[bytes]:
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_name)
        return response["Body"].read()
    except ClientError as e:
        logger.error("MinIO get failed: %s", e)
        return None

def delete_from_minio(object_name: str) -> bool:
    s3 = get_s3_client()
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=object_name)
        return True
    except ClientError as e:
        logger.error("MinIO delete failed: %s", e)
        return False
