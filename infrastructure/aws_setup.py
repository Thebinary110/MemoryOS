import boto3
import os
from datetime import datetime

class AWSInfrastructure:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        self.data_bucket = os.getenv('DATA_BUCKET')
        self.model_bucket = os.getenv('MODEL_BUCKET')
    
    def upload_document(self, file_path: str, doc_id: str):
        key = f"documents/{doc_id}/{os.path.basename(file_path)}"
        self.s3.upload_file(file_path, self.data_bucket, key)
        return f"s3://{self.data_bucket}/{key}"
    
    def log_metric(self, metric_name: str, value: float, unit: str = 'Count'):
        self.cloudwatch.put_metric_data(
            Namespace='MemoryHackathon',
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit,
                    'Timestamp': datetime.utcnow()
                }
            ]
        )

_aws = None

def get_aws():
    global _aws
    if _aws is None:
        _aws = AWSInfrastructure()
    return _aws
