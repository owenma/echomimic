import os
import shutil
import time
from tempfile import NamedTemporaryFile
from utils.imageUtil import save_image
import redis
import requests

from utils import cosUtil
from utils import RedisLockUtil

host = "118.25.99.26"
port = 6390
pwd = ""
base_path = os.getcwd()
driver_video_path = os.path.join(base_path, "resources/drive_wly.pkl")
print(f"driver video path：{driver_video_path}")
# stream对应的key
req_stream_name = "dhuman:liveportrait:stream"
resp_stream_name = "dhuman:loopmakesucc:stream"
stream_lock = "dhuman:video:lock:key"

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
            new_filepath = f"{tmp_file.name}.jpg"
            tmp_file.close()

            # 图片转为 RGB 模式，保存为 JPEG
            from PIL import Image
            img = Image.open(tmp_file.name)  # 假设你有一个 PNG 图像
            img = img.convert('RGB')  # 转换为 RGB
            img.save(new_filepath, 'JPEG')  # 保存为 JPEG

            # os.rename(tmp_file.name, new_filepath)
            # 图片预处理
            save_image(new_filepath, new_filepath)
            print(f"文件已重命名为： {new_filepath}")
            return os.path.basename(tmp_file.name), new_filepath
        else:
            print(f"Error downloading file: {response.status_code}")
            return


def delete_directory(path: str):
    """删除指定路径的目录及其内容"""
    # 检查路径是否存在
    if os.path.exists(path):
        # 使用shutil.rmtree来删除整个目录
        shutil.rmtree(path)
        print(f"目录 {path} 及其内容已被删除。")
    else:
        print(f"指定的路径 {path} 不存在。")


def process():
    messages = rds.xread({req_stream_name: '0'}, block=1)  # Block until a new message arrives
    print("start live: ")
    for stream, message_list in messages:
        for message_id, msgObj in message_list:
            lock = RedisLockUtil.RedisLock(rds, stream_lock)
            if lock.acquire_lock():
                try:
                    # Process the message
                    print(f"Received message: {msgObj}")
                    start = int(time.time())
                    print("start time: ", start)
                    # Acknowledge the message by removing it from the stream
                    # msgObj = json.loads(json.dump(message_data))
                    uid = msgObj["uid"]
                    picture = msgObj["picture"]
                    dhuman_id = msgObj["remember_id"]
                    # 设置时间
                    rds.set(uid + "_" + dhuman_id, str(int(time.time())))
                    cos_key_path = os.path.join("dhuman", uid, dhuman_id, str(int(time.time())))
                    folder_path = os.path.join(base_path, cos_key_path)
                    picture_name, picture_path = download_file(cosUtil.getCacheSignUrl(rds, picture))
                    print(f"原始图片文件名：{picture_name}")
                    print(f"原始图片路径：{picture_path}")
                    print(f"结果存储路径：{folder_path}")
                    cmd = "cd ../LivePortrait && python inference.py -s %s -d %s -o %s --no-flag-stitching --flag_crop_driving_video" % (
                        picture_path, driver_video_path, folder_path)
                    print(cmd)
                    os.system(cmd)

                    # 本地路径
                    result_file = os.path.join(folder_path, f"{picture_name}--drive_wly.mp4")
                    # 上传COS路径
                    cos_key_file = os.path.join(cos_key_path, f"{picture_name}--drive_wly.mp4")

                    print("start upload file:", result_file)
                    cosUtil.uploadfile(cos_key_file, result_file)
                    print(f"上传COS路径：{cos_key_file}")
                    ret_url = cosUtil.getfileurl(cos_key_file)
                    # 删除本地文件
                    delete_directory(folder_path)
                    # 通知业务系统（dhuman）
                    respObj = {
                        "remember_id": dhuman_id,
                        "video": ret_url
                    }
                    rds.xadd(resp_stream_name, respObj)
                    print(f"发送redis消息成功！队列：{resp_stream_name}；消息ID：{respObj}")

                    end = int(time.time())
                    print("end time: ", end)
                    print("use time: ", end - start)
                    print(rds.get(uid + "_" + dhuman_id))
                    rds.xdel(req_stream_name, message_id)
                    print(f"删除已完成队列消息：{req_stream_name}；消息ID：{message_id}")
                except Exception as e:
                    print("出现异常：", e)
                finally:
                    rds.xdel(req_stream_name, message_id)
                    lock.release_lock()
