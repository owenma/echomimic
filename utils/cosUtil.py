# -*- coding=utf-8
import time

from qcloud_cos import CosConfig
from urllib.parse import unquote, urlparse
from qcloud_cos import CosS3Client
from qcloud_cos.cos_exception import CosClientError, CosServiceError
import sys
import os
import logging

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# 1. 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在 CosConfig 中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = ''
secret_key = ''
region = 'ap-shanghai'      # 替换为用户的 region，已创建桶归属的region可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
                           # COS 支持的所有 region 列表参见 https://cloud.tencent.com/document/product/436/6224
token = None               # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
client = CosS3Client(config)

def downloadfile(key = "", folder = "", filename ="test.mp4"):
    ####  获取文件到本地
    response = client.get_object(
        Bucket='dhuman-1323411073',
        Key=key,
    )
    response['Body'].get_stream_to_file(folder+filename)
def outputfile(key = "", uid = 0, chat_id = ""):
    ####  获取文件到本地
    response = client.get_object(
        Bucket='dhuman-1323411073',
        Key=key,
    )
    response['Body'].get_stream_to_file('result/'+uid+"_"+chat_id+'.mp4')

def uploadfile(key, localpath = ""):
    client.put_object_from_local_file(
        Bucket='dhuman-1323411073',
        LocalFilePath=localpath,
        Key=key
    )


def getfileurl(key):
    return client.get_object_url(
        Bucket='dhuman-1323411073',
        Key=key,
    )

def getSignUrl(key):
    return client.get_presigned_url(
        Method='GET',
        Bucket='dhuman-1323411073',
        Key=key,
        Expired=3600  # 3600秒后过期，过期时间请根据自身场景定义
    )

def getCacheSignUrl(rds,url):
    val = rds.get(url)
    if val is not None and len(val)>0:
        return val
    key = urlparse(url).path.split(".com")[-1]
    url_sign = unquote(getSignUrl(key))
    # rds.setex(url,  3000, url_sign)
    return url_sign

if __name__ == '__main__':
    print(int(time.time()))
