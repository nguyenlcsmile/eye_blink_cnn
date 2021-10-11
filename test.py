import tensorflow as tf 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import solver 
import Blink_CNN
import process_untils
import plot 
import gen_video
import func_math
import model_utils
from tqdm import tqdm

tf.compat.v1.disable_eager_execution() #Tránh xung đột giữa tensorflow 2 và 1 trong quá trình tính toán

input_path_video = r'video_test_nguyen.mp4'
imgs, frame_img, fps, width, height = process_untils.eda_video(f'{input_path_video}')

#Size image 
factor = float(300) / height
#Lấy ảnh mắt trái, phải và ảnh resize  
left_eyes = []
right_eyes = []
align_imgs = []
#Load model dlib
model_dlib = model_utils.model_dlib('dlib_model/shape_predictor_68_face_landmarks.dat')

print("[INFO] Computing face detections...")
for i, img in enumerate(tqdm(imgs)):
    #Chuyển ảnh sang gray 
    face_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Lấy tọa độ và crop ảnh khuôn mặt trong ảnh 
    face_ache, faces = process_untils.align(img, model_dlib, func_math.lamdmard_2D())
    # Nhan dien cac diem landmark
    # Tao mot hinh chu nhat quanh khuon mat
    
    if len(face_ache) == 0:
        left_eyes.append(None)
        right_eyes.append(None)
        continue

    aligned_cur_im, aligned_cur_shapes = process_untils.get_aligned_face_and_landmarks(img, face_ache)
    # crop eyes
    leye, reye = process_untils.crop_eye(aligned_cur_im[0], aligned_cur_shapes[0])
    im_resized = cv2.resize(img, None, None, fx=factor, fy=factor)

    left_eyes.append(leye)
    right_eyes.append(reye)
    align_imgs.append(im_resized)

print(len(imgs), len(left_eyes), len(right_eyes), len(align_imgs), frame_img)
out_dir = 'cnn_test/'
#Build network 
net = Blink_CNN.BlinkCNN(is_train=False)
net.build()

sess = tf.compat.v1.Session()
#Init solver
solver = solver.Solver(sess=sess, net=net)
solver.init()

total_eye1_prob = []
total_eye2_prob = []
plot_vis_list = []

if frame_img != len(imgs):
    frame_img = len(imgs)

print("[INFO] Computing eyes recognize...")
for i in range(frame_img):
    # print('Frame: ' + str(i))
    eye1, eye2 = left_eyes[i], right_eyes[i]
    if eye1 is not None:
        eye1_prob, = solver.test([eye1])
    else:
        eye1_prob = 0.5

    if eye2 is not None:
        eye2_prob, = solver.test([eye2])
    else:
        eye2_prob = 0.5
    
    total_eye1_prob.append(eye1_prob[0, 1])
    total_eye2_prob.append(eye2_prob[0, 1])
    plot_vis_list.append(plot.plot_video(frame_img, fps, i, total_eye1_prob, total_eye2_prob))

sess.close()
tf.compat.v1.reset_default_graph()
gen_video.gen_videos(input_path_video, frame_img, align_imgs, fps, out_dir, plot_vis_list, tag='')


