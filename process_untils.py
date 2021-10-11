import cv2 
import os 
import numpy as np 
import tensorflow as tf 
import model_utils
import func_math

def eda_video(path_video):
    '''
        Hàm để lấy các tính chất của một video như: FPS, width, height, số lượng frame của video
        Đầu vào: Đường dẫn video
        Đầu ra: các tính chất của video
    '''
    #Đọc video
    vidcap = cv2.VideoCapture(path_video)
    #Lấy số lượng image trong video
    frame_img = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Tính FPS của video
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    #Lấy kích thước của video
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #Lấy toàn bộ ảnh trong video
    imgs = []
    while True:
        ret, frame = vidcap.read()
        if ret:
            imgs.append(frame)
        else:
            break

    vidcap.release()
    #Kiểm tra số lượng image trong video
    if len(imgs) != frame_img:
        frame_img = len(imgs)
        
    return imgs, frame_img, fps, width, height

def align(image, lmark_predictor, landmarks_2D):
    ''' 
        Lấy tất cả các khuôn mặt trong video, image và matrix translate điều chỉnh tọa độ mắt ngang nhau 
        và lấy ảnh mắt trái và phải 
    '''

    #Compute rectangle faces in images
    faces = model_utils.model_detection_faces(image)

    face_list = []
    # print("Face:", faces)
    if faces is not None or len(faces) > 0:
        for pred in faces:
            points = func_math.shape_to_np(lmark_predictor(image, pred))
            # print("Points:", len(points))
            # print(landmarks_2D.shape, points[17:].shape)
            trans_matrix = func_math.umeyama(points[17:], landmarks_2D, True)[0:2]
            # print("Trans_matrix:", trans_matrix)
            face_list.append([trans_matrix, points])
            # print("Face_list:", face_list)

    return face_list, faces

def get_2d_aligned_face(image, mat, size=256, padding=[0, 0]):
    ''' 
        Lấy tất cả các khuôn mặt trong image bằng cách trượt và scale qua image
        giống như kiểu xoay ảnh(rotation),... bằng matrix transate,..
        Đầu vào: 
         --Image: ảnh gốc
         --Mat: matrix trans/scale được tìm bằng thuật toán least square estimattion bằng hàm umeyama
         ở trên 
         -- size: kích thước khuôn mặt mình muốn (mình tự định nghĩa)
        Đầu ra: Khuôn mặt được scale và 2 con mắt có tọa độ y bằng nhau(ngang nhau) nhờ matrix trans tìm được ở umeyama.
    '''
    mat = mat * size
    mat[0, 2] += padding[0]
    mat[1, 2] += padding[1]

    return cv2.warpAffine(image, mat, (size + 2 * padding[0], size + 2 * padding[1]))

def get_aligned_face_and_landmarks(image, face_list, aligned_face_size = 256, padding=(0, 0)):
    """
    get all aligned faces and landmarks of all images
    :param imgs: origin images
    :param fa: face_alignment package
    ==================
    :return: trả về ảnh khuôn mặt yêu cầu và ma trận để tính toán thu được ảnh mắt 
    """
    aligned_cur_shapes = []
    aligned_cur_im = []
    for mat, points in face_list:
        # Get transform matrix (thu được hình ảnh khuôn mặt)
        aligned_face = get_2d_aligned_face(image, mat, aligned_face_size, padding)
        # Mapping landmarks to aligned face
        pred_ = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        pred_ = np.transpose(pred_)
        mat = mat * aligned_face_size
        mat[0, 2] += padding[0]
        mat[1, 2] += padding[1]
        aligned_pred = np.dot(mat, pred_)
        aligned_pred = np.transpose(aligned_pred[:2, :])
        aligned_cur_shapes.append(aligned_pred)
        aligned_cur_im.append(aligned_face)

    return aligned_cur_im, aligned_cur_shapes 

def crop_eye(image, points):
    '''
        Tính toán lấy ảnh mắt trái và phải của face
        Input: image chứa face, tọa độ các điểm đặc biệt trên khuôn mặt
        Output: ảnh mắt trái và phải của khuôn mặt
    '''
    eyes_list = []

    left_eye = points[36:42, :]
    right_eye = points[42:48, :]
    eyes = [left_eye, right_eye]

    for j in range(len(eyes)):
        lp = np.min(eyes[j][:, 0])
        rp = np.max(eyes[j][:, 0])
        tp = np.min(eyes[j][:, -1])
        bp = np.max(eyes[j][:, -1])

        w = rp - lp
        h = bp - tp

        lp_ = int(np.maximum(0, lp - 0.25 * w))
        rp_ = int(np.minimum(image.shape[1], rp + 0.25 * w))
        tp_ = int(np.maximum(0, tp - 1.75 * h))
        bp_ = int(np.minimum(image.shape[0], bp + 1.75 * h))

        eyes_list.append(image[tp_:bp_, lp_:rp_, :])

    return eyes_list