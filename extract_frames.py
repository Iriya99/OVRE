import os
import sys
from tqdm import tqdm

start = int(sys.argv[1])
end = int(sys.argv[2])

src = 'Your Video Directory'
dst = 'Your Frame Directory'

def load_frames_from_video(src, dst):
    cmd = 'ffmpeg -i '+ src +' -q 0 -r 30 '+ dst + '/%05d.jpg ' + '-loglevel quiet'
    os.system(cmd)

videos = os.listdir(src)

for i in tqdm(range(len(videos))):
    os.mkdir(dst + videos[i].split('.')[0])
    load_frames_from_video(src + videos[i], dst + videos[i].split('.')[0])

# # Process short videos
# import os
# from utils.basic_utils import deletedir

# short_videos = []
# cnt = 0
# vlist = os.listdir(dst)
# for i in range(len(vlist)):
#     while len(os.listdir(dst + vlist[i])) < 20:
#         imgs = sorted(os.listdir(dst + vlist[i]))
#         idx = int(imgs[-1].split('.')[0]) + 1
#         fname = '{:05d}.jpg'.format(idx)
#         print(fname)
    
#         cmd = 'cp ' + dst + vlist[i] + '/' + imgs[-1] + ' ' + dst + vlist[i] + '/' + fname
#         os.system(cmd)

# for i in range(len(short_videos)):
#     deletedir('./data/images/' + short_videos[i])
#     os.mkdir('./data/images/' + short_videos[i])
#     load_frames_from_video(src + short_videos[i] + '.mp4', dst)