import cv2 
import numpy as np 
import os 
from imutils import face_utils
from tqdm import tqdm
import pathlib
import process_untils 
import func_math 
import model_utils

#Tính khoảng cách giữa 2 điểm trong không gian nhiều chiều
def e_dist(pA, pB):
	return np.linalg.norm(pA - pB)

def eye_ratio(eye):
	#Tính toán khoảng cách theo chiều dọc giữa mi trên và mi dưới của mắt 
	d_V1 = e_dist(eye[1], eye[5])
	d_V2 = e_dist(eye[2], eye[4])

	#Tính khoảng cách theo chiều ngang giữa 2 đuôi mắt
	d_H = e_dist(eye[0], eye[3])

	#Tính tỉ lệ giữa chiều ngang và chiều dọc
	eye_ratio_val = (d_V1 + d_V2) / (2.0 * d_H)

	return eye_ratio_val

def label_images(path_data, path_eyes, landmarks_2D=func_math.lamdmard_2D()):
    ''' 
        Dùng để lable image mắt mở(1), đóng(0) bằng cách lấy tỉ lệ chiều dọc và ngang của
        mắt sau khi tính toán ... nếu tỉ lệ lớn hơn ngưỡng (threshold) thì cho là mở và ngược lại.
        
        --path_data: path chứa video
        --path_eye: path chứa image mắt trái và mắt phải sau khi crop 
    '''
    #File chứa path ảnh và label ảnh 
    path_data_npy = 'data_train.npy'
    list_path_eye = []
    label_eye = []

    #Create folder contain image eyes 
    if not os.path.exists(path_eyes):
        os.mkdir(path_eyes)
    
    #Lấy tất cả các video trong datasets
    path_folder = pathlib.Path(path_data)
    path_videos = sorted(list(path_folder.glob("*mp4")))

    model_dlib = model_utils.model_dlib('dlib_model/shape_predictor_68_face_landmarks.dat')
    number_image = 0

    #Lấy danh sách các cụm điểm landmark cho 2 mắt
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    #Xử lí và label các ảnh mắt đóng và mở 
    for path in path_videos[:50]:
        # print(path)
        imgs, frame_img, fps, width, height = process_untils.eda_video(f'{path}')
        #Size image 
        factor = float(300) / height
        #Lấy ảnh mắt trái, phải và ảnh resize  
        left_eyes = []
        right_eyes = []
        align_imgs = []
        
        for i, img in enumerate(tqdm(imgs)):
            #Chuyển ảnh sang gray 
            face_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            #Lấy tọa độ và crop ảnh khuôn mặt trong ảnh 
            face_ache, faces = process_untils.align(img, model_dlib, landmarks_2D)

            #Nhận diện các điểm landmark
            #Tạo một hình chữ nhật bao quanh khuôn mặt
            if len(faces) > 0:
                for j in range(len(faces)):
                    landmark = model_dlib(face_gray, faces[j])
                    landmark = face_utils.shape_to_np(landmark)

                    #Lấy tọa độ ảnh mắt trái và phải (khung chứa ảnh trái và phải)
                    leftEye = landmark[left_eye_start:left_eye_end]
                    rightEye = landmark[right_eye_start:right_eye_end]

                    #Tính tỷ lệ mắt trái, phải để xác định mắt mở hay đóng để label 
                    left_eye_ratio = eye_ratio(leftEye)
                    right_eye_ratio = eye_ratio(rightEye)
                    # print(left_eye_ratio, right_eye_ratio)
                
                    if len(face_ache) == 0:
                        left_eyes.append(None)
                        right_eyes.append(None)
                        continue

                    aligned_cur_im, aligned_cur_shapes = process_untils.get_aligned_face_and_landmarks(img, face_ache)
                    # crop eyes
                    leye, reye = process_untils.crop_eye(aligned_cur_im[j], aligned_cur_shapes[j])
                    im_resized = cv2.resize(img, None, None, fx=factor, fy=factor)

                    cv2.imwrite(path_eyes + '{:09n}'.format(number_image) + '.png', leye)
                    list_path_eye.append(path_eyes + '{:09n}'.format(number_image) + '.png')
                    cv2.imwrite(path_eyes + '{:09n}'.format(number_image + 1) + '.png', reye)
                    list_path_eye.append(path_eyes + '{:09n}'.format(number_image + 1) + '.png')
                    label_eye.append([1 if left_eye_ratio > 0.25 else 0])
                    label_eye.append([1 if right_eye_ratio > 0.25 else 0])

                    #Kiểm tra quá trình label
                    if len(label_eye) % 200 == 0:
                        print(list_path_eye[len(label_eye) - i], label_eye[len(label_eye) - i])

                    number_image += 2
                    left_eyes.append(leye)
                    right_eyes.append(reye)
                    align_imgs.append(im_resized)

    #Save data with numpy file npy
    data = [i for i in zip(list_path_eye, label_eye)]
    np.save(path_data_npy, data)
    
    return