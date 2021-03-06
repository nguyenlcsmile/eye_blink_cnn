import numpy as np 

def shape_to_np(shape, dtype="int"):
    ''' 
        Lấy tọa độ các điểm trên khuôn mặt lưu dưới dạng tuple và lưu vào một np.array
        Đầu vào: list tọa độ các điểm predict
        Đầu ra: array các tuple của các điểm predict, mỗi tuple có 2 elements đại diện cho x, y (chính là from imutils import face_utils.shape_to_np)
                tổng cộng có 68 các đối tượng trên khuôn mặt được đại diện
    '''
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def umeyama(src, dst, estimate_scale):
    ''' 
        umeyama function from scikit-image/skimage/transform/_geometric.py

        Estimate N-D similarity transformation with or without scaling.
        Parameters
        ----------
        src : (M, N) array
            Source coordinates.
        dst : (M, N) array
            Destination coordinates.
        estimate_scale : bool
            Whether to estimate scaling factor.
        Returns
        -------
        T : (N + 1, N + 1)
            The homogeneous similarity transformation matrix. The matrix contains
            NaN values only if the problem is not well-conditioned.
        
        References
        ----------
        .. [1] "Least-squares estimation of transformation parameters between two
                point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
        '''

    num = src.shape[0]
    dim = src.shape[1]

    #Compute mean of src and dst 
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    #Subtract mean from src and dst
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T

def lamdmard_2D():
    #Tọa độ các điểm như mắt, mũi, miệng,... trên khuôn mặt do người nghiên cứu tìm ra 
    mean_face_x = np.array([
        0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
        0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
        0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
        0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
        0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
        0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
        0.553364, 0.490127, 0.42689])

    mean_face_y = np.array([
        0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
        0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
        0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
        0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
        0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
        0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
        0.784792, 0.824182, 0.831803, 0.824182])
    
    return np.stack([mean_face_x, mean_face_y], axis=1)
