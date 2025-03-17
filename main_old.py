from time import strftime
import os, sys, time
import redis, logging
from utils import cosUtil
import urllib.parse
import requests, json
from tempfile import NamedTemporaryFile
from IPython.display import display, Video

from mzq_infer_audio2vid import load_config, load_models, infer

# --------------------------------------------------------------------------------

os.chdir('/root/EchoMimic')
config = load_config()
models = load_models(
    config=config,
)

print('\n模型加载完毕！\n')

host = "118.25.99.26"
port = 6390
pwd = "dhuman2024"
# stream对应的key
req_stream_name = "dhuman:video:mixture:queue"
resp_stream_name = "dhuman:video:makesucc:queue"

# 实现一个生产者
rds = redis.StrictRedis(host=host, port=port, db=0, decode_responses=True)


def download_file(url):
    with NamedTemporaryFile(mode='wb', delete=False) as tmp_file:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # 将响应内容写入临时文件
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            print(f"文件已保存为 {tmp_file.name}")
            return tmp_file
        else:
            print(f"Error downloading file: {response.status_code}")
            return


width = 512
height = 512
video_length = 2160
seed = 32
facemusk_dilation_ratio = 0.1
facecrop_dilation_ratio = 0.5
context_frames = 12
context_overlap = 3
cfg = 2.5
steps = 30
sample_rate = 16000
fps = 24

if __name__ == '__main__':
    while True:
        msg = rds.lpop(req_stream_name)  # Block until a new message arrives
        if msg:
            try:
                # Process the message
                msgObj = json.loads(msg)
                print(f"Received message: {msgObj}")
                start = int(time.time())
                print("start time: ", start)
                # Acknowledge the message by removing it from the stream
                # msgObj = json.loads(json.dump(message_data))
                uid = str(msgObj["uid"])
                video = msgObj["video"]
                audio = msgObj["audio"]
                dhuman_id = str(msgObj["remember_id"])
                chat_id = str(msgObj["chat_id"])
                video_id = str(msgObj["video_id"])
                # 设置时间
                rds.set(uid + "_" + chat_id, str(int(time.time())))
                folder_path = "/root/autodl-tmp/dhuman/" + uid + "/" + dhuman_id

                video_file = download_file(cosUtil.getCacheSignUrl(rds, video))
                audio_file = download_file(cosUtil.getCacheSignUrl(rds, audio))
                video_path_name = video_file.name
                audio_path_name = audio_file.name
                print(video_path_name)
                print(audio_path_name)
                assert os.path.exists(video_path_name)
                assert os.path.exists(audio_path_name)

                config.test_cases = {
                    video_path_name: [audio_path_name],
                }
                start = time.time()
                path_video = infer(
                    models=models,
                    config=config,
                    width=width,
                    height=height,
                    video_length=video_length,
                    seed=seed,
                    facemusk_dilation_ratio=facemusk_dilation_ratio,
                    facecrop_dilation_ratio=facecrop_dilation_ratio,
                    context_frames=context_frames,
                    context_overlap=context_overlap,
                    cfg=cfg,
                    steps=steps,
                    sample_rate=sample_rate,
                    fps=fps,
                    ret_path=folder_path,
                )
                end = time.time()
                print(f'\n视频生成完毕！\n耗时：{end - start:.3f} s.\n')

                print("start upload file:", path_video)
                key = path_video[len("/root/autodl-tmp/"):]
                cosUtil.uploadfile(key, path_video)
                ret_url = cosUtil.getfileurl(key)
                print(ret_url)
                respObj = {
                    "chat_id": chat_id,
                    "video": ret_url,
                    "status": "1",
                    "video_id": video_id,
                }
                rds.lpush(resp_stream_name, json.dumps(respObj))

                end = int(time.time())
                print("end time: ", end)
                print("use time: ", end - start)
                print(rds.get(uid + "_" + chat_id))
                video_file.close()
                audio_file.close()
            except Exception as e:
                respObj = {
                    "chat_id": chat_id,
                    "video": ret_url,
                    "video_id": video_id,
                    "status": "-1",
                }
                rds.lpush(resp_stream_name, json.dumps(respObj))
                print("出现异常：", e)

        time.sleep(1)
