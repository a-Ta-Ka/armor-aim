#!/usr/bin/env python3       
import os                  #导入标准库
import rosbag              #导入ros1的py包 

BAG = '/root/bags/red_4.bag'                   #读取包的绝对路径
TOPIC = '/hk_camera/image_raw/compressed'       #读取话题名
OUT_DIR = '/root/vision_ws/frames'              #输出目录
NUM = 10                                        #要提取的帧数

os.makedirs(OUT_DIR, exist_ok=True)             #创造输出目录

bag = rosbag.Bag(BAG)                           #只读开包
msgs = list(bag.read_messages(topics=[TOPIC]))  
# 最里面函数返回生成器，参数指定包、话题，list转为列表，存进msgs中
bag.close()                                     #关闭包，释放句柄

print(f'{BAG} 共 {len(msgs)} 帧')
for k in range(NUM):
    _, msg, _ = msgs[k * len(msgs) // NUM]      #筛选
    path = os.path.join(OUT_DIR, f'frame_{k+10:03d}.jpg') #拼接输出路径
    with open(path, 'wb') as f:
        f.write(msg.data)                       # 把帧数据存成jpg文件
    print(path)

# cd /root/vision_ws
# python3 scripts/extract_frames.py