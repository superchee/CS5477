import numpy as np
import scipy.io as sio
import h5py
import cv2
import matplotlib.pyplot as plt


"""Helper functions: You should not have to touch the following functions.
"""
def compute_right_epipole(F):

    U, S, V = np.linalg.svd(F.T)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(img1, img2, F, x1, x2, epipole=None, show_epipole=False):
    """
    Visualize epipolar lines in the imame

    Args:
        img1, img2: two images from different views
        F: fundamental matrix
        x1, x2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
    Returns:

    """
    plt.figure()
    plt.imshow(img1)
    for i in range(x1.shape[1]):
      plt.plot(x1[0, i], x1[1, i], 'bo')
      m, n = img1.shape[:2]
      line1 = np.dot(F.T, x2[:, i])
      t = np.linspace(0, n, 100)
      lt1 = np.array([(line1[2] + line1[0] * tt) / (-line1[1]) for tt in t])
      ndx = (lt1 >= 0) & (lt1 < m)
      plt.plot(t[ndx], lt1[ndx], linewidth=2)
    plt.figure()
    plt.imshow(img2)

    for i in range(x2.shape[1]):
      plt.plot(x2[0, i], x2[1, i], 'ro')
      if show_epipole:
        if epipole is None:
          epipole = compute_right_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


      m, n = img2.shape[:2]
      line2 = np.dot(F, x1[:, i])

      t = np.linspace(0, n, 100)
      lt2 = np.array([(line2[2] + line2[0] * tt) / (-line2[1]) for tt in t])

      ndx = (lt2 >= 0) & (lt2 < m)
      plt.plot(t[ndx], lt2[ndx], linewidth=2)
    plt.show()


def compute_essential(data1, data2, K):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
        K: intrinsic matrix of the camera
    Returns:
        E: Essential matrix
    """

    """YOUR CODE STARTS HERE"""
    #E, _ = cv2.cv2.findEssentialMat(data1[:2, :].T, data2[:2, :].T, cameraMatrix=K)
    # src = data1[0:2,:].T.copy()
    # dst = data2[0:2,:].T.copy()

    #number of points
    N = data1.shape[1]


    #Normalization
    # m_src = src.mean(0)
    # md_src = ((((src - m_src)**2).sum(1))**(1/2)).mean() #mean distance of all points from centroid
    # s_src = np.sqrt(2)/md_src
    # m_dst = dst.mean(0)
    # md_dst = ((((dst - m_dst)**2).sum(1))**(1/2)).mean()
    # s_dst = np.sqrt(2)/md_dst

    # S_tr_src = np.array([[s_src,0,-s_src*m_src[0]],[0,s_src,-s_src*m_src[1]],[0,0,1]])
    # S_tr_dst = np.array([[s_dst,0,-s_dst*m_dst[0]],[0,s_dst,-s_dst*m_dst[1]],[0,0,1]])

    # src_tr_points = np.dot(S_tr_src, np.concatenate((src.T, np.ones((1,N)))))
    # dst_tr_points = np.dot(S_tr_dst, np.concatenate((dst.T, np.ones((1,N)))))

    # norm_src_K = np.linalg.pinv(K) @ src_tr_points
    # norm_dst_K = np.linalg.pinv(K) @ dst_tr_points

    norm_src_K = np.linalg.pinv(K) @ data1
    norm_dst_K = np.linalg.pinv(K) @ data2

    norm_src = norm_src_K.T
    norm_dst = norm_dst_K.T


    #A matrix
    A = []
    for i in range(N):
        x,y = norm_src[i,0], norm_src[i,1]
        x1,y1 = norm_dst[i,0], norm_dst[i,1]
        A.append([x1*x,x1*y,x1,y1*x,y1*y,y1,x,y,1])
    
    A = np.asarray(A)
    
    #SVD of A
    
    U, S, V = np.linalg.svd(A)


    #take the last column of V.T
    h = V[-1,:]
    E1 = h.reshape(3,3)

    E_U, E_S, E_V = np.linalg.svd(E1)

    E = E_U @ np.diag([(E_S[0]+E_S[1])/2,(E_S[0]+E_S[1])/2,0]) @ E_V

    #E = S_tr_dst.T @ E @ S_tr_src
    # E = E/E[-1,-1]   
    
    """YOUR CODE ENDS HERE"""

    return E


def decompose_e(E, K, data1, data2):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        E: Essential matrix
        K: intrinsic matrix of the camera
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        trans: 3x4 array representing the transformation matrix
    """
    """YOUR CODE STARTS HERE"""
    #_, r, t, _ = cv2.recoverPose(E, data1[:2, :].T, data2[:2, :].T, K)
    #trans = np.concatenate([r, t], axis =1)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    U, S, VT= np.linalg.svd(E)

    R_check = U @ W @ VT
    if np.linalg.det(R_check) < 0:
        U, S, VT = np.linalg.svd(-E)

    T1 = U[:,-1]
    T2 = -U[:,-1]

    R1 = U @ W @ VT
    R2 = U @ W.T @ VT


    P = K @ np.concatenate([np.eye(3),np.zeros([3,1])],axis=1)


    trans_all = []
    trans_all.append(np.concatenate([R1,T1.reshape(3,1)],axis=1))
    trans_all.append(np.concatenate([R1,T2.reshape(3,1)],axis=1))
    trans_all.append(np.concatenate([R2,T1.reshape(3,1)],axis=1))
    trans_all.append(np.concatenate([R2,T2.reshape(3,1)],axis=1))


    max_points_num = 0
    for trans_cur in trans_all:
        P1 = K @ trans_cur
        points_num = 0
        for x1, x2 in zip(data1.T,data2.T):
            A = []
            A.append(x1[0] * P[2,:] - P[0,:])
            A.append(x1[1] * P[2,:] - P[1,:])
            A.append(x2[0] * P1[2,:] - P1[0,:])
            A.append(x2[1] * P1[2,:] - P1[1,:])
            A = np.array(A)
    
            U,S,VH = np.linalg.svd(A)
            x = VH[-1,:]
            x = x/x[-1]    

            x_3d = x[0:3]

            z1 = trans_cur[2,0:3] @ (x_3d - trans_cur[:,3])
            z2 = x_3d[2]#np.array([0,0,1]) @ (x_3d - np.array([0,0,0]))
            if z1 > 0 and z2 > 0:
                points_num = points_num+1
        if points_num > max_points_num:
            trans = trans_cur
            max_points_num = points_num

    """YOUR CODE ENDS HERE"""
    return trans


def compute_fundamental(data1, data2):
    """
    Compute the fundamental matrix from point correspondences

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        F: fundamental matrix
    """

    """YOUR CODE STARTS HERE"""
    #F, _ = cv2.findFundamentalMat(data1[:2, :].T, data2[:2, :].T, method = cv2.FM_8POINT)
    src = data1[0:2,:].T.copy()
    dst = data2[0:2,:].T.copy()

    #number of points
    N = src.shape[0]


    #Normalization
    m_src = src.mean(0)
    md_src = ((((src - m_src)**2).sum(1))**(1/2)).mean() #mean distance of all points from centroid
    s_src = np.sqrt(2)/md_src
    m_dst = dst.mean(0)
    md_dst = ((((dst - m_dst)**2).sum(1))**(1/2)).mean()
    s_dst = np.sqrt(2)/md_dst

    S_tr_src = np.array([[s_src,0,-s_src*m_src[0]],[0,s_src,-s_src*m_src[1]],[0,0,1]])
    S_tr_dst = np.array([[s_dst,0,-s_dst*m_dst[0]],[0,s_dst,-s_dst*m_dst[1]],[0,0,1]])

    src_tr_points = np.dot(S_tr_src, np.concatenate((src.T, np.ones((1,N)))))
    dst_tr_points = np.dot(S_tr_dst, np.concatenate((dst.T, np.ones((1,N)))))

    norm_src = src_tr_points.T
    norm_dst = dst_tr_points.T

    #A matrix
    A = []
    for i in range(N):
        x,y = norm_src[i,0], norm_src[i,1]
        x1,y1 = norm_dst[i,0], norm_dst[i,1]
        A.append([x1*x,x1*y,x1,y1*x,y1*y,y1,x,y,1])
    
    A = np.asarray(A)
    
    #SVD of A
    
    U, S, V = np.linalg.svd(A)


    #take the last column of V.T
    h = V[-1,:]
    F1 = h.reshape(3,3)

    F_U, F_S, F_V = np.linalg.svd(F1)

    F = F_U @ np.diag([F_S[0],F_S[1],0]) @ F_V

    F = S_tr_dst.T @ F @ S_tr_src
    F = F/F[-1,-1]
    
    """YOUR CODE ENDS HERE"""

    return F











