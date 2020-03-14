""" CS4277/CS5477 Lab 2: Camera Calibration.
See accompanying Jupyter notebook (lab2.ipynb) for instructions.

Name: CAOQI
Email: e0338189@u.nus.edu
NUSNET ID: e0338189
"""




import cv2
import numpy as np
from scipy.optimize import least_squares

"""Helper functions: You should not have to touch the following functions.
"""

def convt2rotation(Q):
    """Convert a 3x3 matrix into a rotation matrix

    Args:
        Q (np.ndarray): Input matrix

    Returns:
        R (np.ndarray): A matrix that satisfies the property of a rotation matrix
    """

    u,s,vt = np.linalg.svd(Q)
    R = np.dot(u, vt)

    return R

def vector2matrix(S):
    """Convert the vector representation to rotation matrix,
       You will use it in the error function because the input parameters is in vector format

    Args:
        S (np.ndarray): vector representation of rotation (3,)

    Returns:
        R (np.ndarray): Rotation matrix (3, 3)
    """

    S = np.expand_dims(S, axis=1)
    den = 1 + np.dot(S.T, S)
    num = (1 - np.dot(S.T, S))*(np.eye(3)) + 2 * skew(S) + 2 * np.dot(S, S.T)
    R = num/den
    homo = np.zeros([3,1], dtype=np.float32)
    R = np.hstack([R, homo])
    return R

def skew(a):
    s = np.array([[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]])
    return s
def matrix2quaternion(T):

    R = T[:3, :3]

    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def matrix2vector(R):
    """Convert a rotation matrix into vector representation.
       You will use it to convert a rotation matrix into a vector representation before you pass the parameters into the error function.

    Args:
        R (np.ndarray): Rotation matrix (3, 3)
    Returns:
        Q (np.ndarray): vector representation of rotation (3,)
    """

    Q = matrix2quaternion(R)
    S = Q[1:]/Q[0]
    return S





"""Functions to be implemented
"""

def init_param(pts_model, pts_2d):
    """ Estimate the intrisics and extrinsics of cameras

    Args:
        pts_model (np.ndarray): Coordinates of points in 3D (2, N)
        pts_2d (list): Coordinates of points in 2D, the list includes 2D coordinates in three views 3 * (2, N)

    Returns:
        R_all (list): a list including three rotation matrix
        T_all (list): a list including three translation vector
        K (np.ndarray): a list includes five intrinsic parameters (5,)

    Prohibited functions:
        cv2.calibrateCamera()

    """


    R_all = []
    T_all = []
    K = None
    V_list = []
    H1 = []
    H2 = []
    H3 = []
    for i in range(len(pts_2d)):
        pts_src = pts_model.T
        pts_dst = pts_2d[i].T

        """ YOUR CODE STARTS HERE """
        #ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(pts_src, pts_dst, (640, 480), None, None)
        H,_ = cv2.findHomography(pts_src, pts_dst)
        h1,h2,h3 = H.T

        H1.append(h1)
        H2.append(h2)
        H3.append(h3)

        v_12 = np.array([h1[0]*h2[0],h1[0]*h2[1]+h1[1]*h2[0],h1[1]*h2[1],h1[2]*h2[0]+h1[0]*h2[2],h1[2]*h2[1]+h1[1]*h2[2],h1[2]*h2[2]])
        v_11 = np.array([h1[0]*h1[0],h1[0]*h1[1]+h1[1]*h1[0],h1[1]*h1[1],h1[2]*h1[0]+h1[0]*h1[2],h1[2]*h1[1]+h1[1]*h1[2],h1[2]*h1[2]])
        v_22 = np.array([h2[0]*h2[0],h2[0]*h2[1]+h2[1]*h2[0],h2[1]*h2[1],h2[2]*h2[0]+h2[0]*h2[2],h2[2]*h2[1]+h2[1]*h2[2],h2[2]*h2[2]])

        #V = np.vstack([v_12, v_11-v_22])
        V_list.append(list(v_12))
        V_list.append(list(v_11-v_22))

    V = np.array(V_list)

    _,_,Vh = np.linalg.svd(V)
    b = Vh[-1,:]

    B = [[b[0],b[1],b[3]],[b[1],b[2],b[4]],[b[3],b[4],b[5]]]

    #K_inv = np.linalg.cholesky(B)

    p_y = (B[0][1]*B[0][2]-B[0][0]*B[1][2])/(B[0][0]*B[1][1]-B[0][1]*B[0][1]) #v_0
    lam = B[2][2] - (B[0][2]*B[0][2]+p_y*(B[0][1]*B[0][2]-B[0][0]*B[1][2]))/B[0][0]
    f_x = np.sqrt(lam/B[0][0]) #alpha
    f_y = np.sqrt(lam*B[0][0]/(B[0][0]*B[1][1]-B[0][1]*B[0][1]))#beta
    s = -B[0][1]*f_x*f_x*f_y/lam #gamma
    p_x = s*p_y/f_y - B[0][2]*f_x*f_x/lam #u_0

    K = [f_x,s,p_x,f_y,p_y]

    A = np.array([K[0], K[1], K[2], 0, K[3], K[4], 0, 0, 1]).reshape([3, 3])
    A_inv = np.linalg.inv(A)

    for i in range(len(pts_2d)):
        lam_ex = 1/np.linalg.norm(np.linalg.inv(A)@H1[i])
        r1 = lam_ex * A_inv @ H1[i]
        r2 = lam_ex * A_inv @ H2[i]
        r3 = np.cross(r1,r2)
        t = lam_ex * A_inv @ H3[i]

        R = np.vstack([r1,r2,r3]).T

        R = convt2rotation(R)

        R_all.append(R.copy())
        T_all.append(t.copy())


        """ YOUR CODE ENDS HERE """
    #R_all.append(rvecs)
    #T_all.append(tvecs)
    return R_all, T_all, K



def error_fun(param, pts_model, pts_2d):
    """ Write the error function for least_squares

    Args:
        param (np.ndarray): All parameters need to be optimized. Including intrinsics (0-5), distortion (5-10), extrinsics (10-28).
                            The extrincs consist of three pairs of rotation and translation.

        pts_model (np.ndarray): Coordinates of points in 3D (2, N)
        pts_2d (list): Coordinates of points in 2D, the list includes 2D coordinates in three views 3 * (2, N)

    Returns:
        error : The reprojection error of all points in all three views

    """


    K = param[0:5]
    A = np.array([K[0], K[1], K[2], 0, K[3], K[4], 0, 0, 1]).reshape([3, 3])
    k = param[5:10]
    pts_model_homo = np.concatenate([pts_model, np.ones([1, pts_model.shape[1]])], axis=0)
    points_2d = np.concatenate(pts_2d, axis= 1)
    points_ud_all = []
    for i in range(3):
        s = param[10 + i*6:13+i*6]
        r = vector2matrix(s)
        t = param[13+i*6 : 16+i*6]
        trans = np.array([r[:, 0], r[:, 1], t]).T
        points_ud =  np.dot(trans, pts_model_homo)
        points_ud = points_ud[0:2, :]/points_ud[2:3]
        points_ud_all.append(points_ud)
    points_ud_all = np.concatenate(points_ud_all, axis=1)

    """ YOUR CODE STARTS HERE """
    #points_d = np.zeros_like(points_ud_all)  # replace this line with the real distorted points points_d, where points_d = x_r + dx
    points_d = []
    for j in range(points_ud_all.shape[1]):
        x = points_ud_all[0,j]
        y = points_ud_all[1,j]
        r2 = x*x + y*y
        x_r = (1 + k[0]*r2 + k[1]*r2**2 + k[4]*r2**3)*points_ud_all[:,j]
        d_x = np.array([2*k[2]*x*y + k[3]*(r2+2*x*x),k[2]*(r2+2*y*y)+2*k[3]*x*y])
        x_d = x_r + d_x
        points_d.append(x_d)
    points_d = np.array(points_d).T

    points_d = np.dot(A, np.concatenate([points_d, np.ones([1, points_d.shape[1]])], axis=0))
    points_d = points_d[0:2] / points_d[2:3]

    """ YOUR CODE ENDS HERE """

    error = np.sum(np.square(points_2d - points_d), axis= 0)

    return error



def visualize_distorted(param, pts_model, pts_2d):


    """ Visualize the points after distortion

    Args:
        param (np.ndarray): All parameters need to be optimized. Including intrinsics (0-5), distortion (5-10), extrinsics (10-28).
                            The extrincs consist of three pairs of rotation and translation.

        pts_model (np.ndarray): Coordinates of points in 3D (2, N)
        pts_2d (list): Coordinates of points in 2D, the list includes 2D coordinates in three views 3 * (2, N)

    Returns:
        The visualized results

    """
    K = param[0:5]
    A = np.array([K[0], K[1], K[2], 0, K[3], K[4], 0, 0, 1]).reshape([3, 3])
    k = param[5:10]
    pts_model_homo = np.concatenate([pts_model, np.ones([1, pts_model.shape[1]])], axis=0)
    for i in range(len(pts_2d)):
        s = param[10 + i * 6:13 + i * 6]
        r = vector2matrix(s)
        t = param[13 + i * 6: 16 + i * 6]

        trans = np.array([r[:, 0], r[:, 1], t]).T
        points_ud =  np.dot(trans, pts_model_homo)
        points_ud = points_ud[0:2, :] / points_ud[2:3]

        """ YOUR CODE STARTS HERE """
        #points_d = np.zeros_like(points_ud)  # replace this line with the real distorted points points_d, where points_d = x_r + dx
        points_d = []
        for j in range(points_ud.shape[1]):
            x = points_ud[0,j]
            y = points_ud[1,j]
            r2 = x*x + y*y
            x_r = (1 + k[0]*r2 + k[1]*r2**2 + k[4]*r2**3)*points_ud[:,j]
            d_x = np.array([2*k[2]*x*y + k[3]*(r2+2*x*x),k[2]*(r2+2*y*y)+2*k[3]*x*y])
            x_d = x_r + d_x
            points_d.append(x_d)
        points_d = np.array(points_d).T
        """ YOUR CODE ENDS HERE """



        points_d = np.dot(A, np.concatenate([points_d, np.ones([1, points_d.shape[1]])], axis=0))
        points_d = points_d[0:2] / points_d[2:3]
        points_2d = pts_2d[i]
        img = cv2.imread('./zhang_data/CalibIm{}.tif'.format(i + 1))
        for j in range(points_d.shape[1]):
            cv2.circle(img, (np.int32(points_d[0, j]), np.int32(points_d[1, j])) , 4, (0, 0, 255))
            cv2.circle(img, (np.int32(points_2d[0, j]), np.int32(points_2d[1, j])), 3, (255, 0, 0))
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()





