import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm

def gen_vid(video_path, imgs, fps, width=None, height=None):
    #Combine video
    ext = video_path.split('.')[1]
    if ext == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        video_path = video_path.replace(ext, 'mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    if width is None or height is None:
        height, width = imgs[0].shape[:2]
    else:
        imgs_ = [cv2.resize(img, (width, height)) for img in imgs]
        imgs = imgs_
    
    out = cv2.VideoWriter(video_path, fourcc, fps, (np.int32(width), np.int32(height)))

    for image in imgs:
        out.write(np.uint8(image))
    
    out.release()
    print('The output video is' + video_path)
    return 

def gen_videos(input_path_video, frame_img, align_imgs, fps, out_dir, plot_vis_list, tag=''):
    print(input_path_video)
    vid_name = os.path.basename(input_path_video)
    out_path = os.path.join(out_dir, tag + '_' + vid_name)
    print('Generating video: {}'.format(out_path))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # Out folder
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))
    
    final_list = []

    for i in tqdm(range(frame_img)):
        final_vis = np.concatenate([align_imgs[i], plot_vis_list[i]], axis=1)
        final_list.append(final_vis)
    
    gen_vid(out_path, final_list, fps)
    
    return
