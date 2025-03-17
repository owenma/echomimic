#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/02 7:35 下午
# @FileName: redisService.py

import redis,logging, json
from redis.client import *
import json

host = "118.25.99.26"
port = 6390
pwd = "dhuman2024"
# stream对应的key
req_stream_name = "dhuman:video:mixture:queue"
resp_stream_name = "dhuman:video:makesucc:queue"
# 实现一个生产者
rds = redis.StrictRedis(host=host, port=port, db=0, decode_responses=True)

def producer():
    message_data = {
            "uid": 123,
            "video": "https://dhuman-1323411073.cos.ap-shanghai.myqcloud.com/test/a6.PNG",
            "audio": "https://dhuman-1323411073.cos.ap-shanghai.myqcloud.com/test/yun.wav",
            "remember_id": 70,
            "chat_id": "10",
            "video_id": 11
    }
    rds.rpush(req_stream_name, json.dumps(message_data))
    print(f"发送数据 {message_data}")

    # while True:
    #     messages = rds.xread({stream_name: '0'}, block=0)  # Block until a new message arrives
    #     for stream, message_list in messages:
    #         for message_id, message_data in message_list:
    #             # Process the message
    #             print(f"Received message: {message_data}")
    #             # Acknowledge the message by removing it from the stream
    #             rds.xdel(stream_name, message_id)

if __name__ == "__main__":
    producer()
    u = rds.get("123_chat_id2")
    print(u)