from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time
import random
from copy import deepcopy
from tqdm import tqdm,trange

print("load data ...")
images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")

print("load data done!")

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def mix(n,m):
    result=np.zeros((3,3))
    result[0]=n
    result[1]=m
    result[2]=np.cross(n,m)
    return result.T

def order_eigenvalue(sigmas,E):
    new_sigmas=deepcopy(sigmas)
    new_E=deepcopy(E)
    if abs(new_sigmas[0])<abs(new_sigmas[1]):
        new_sigmas[0]=sigmas[1]
        new_sigmas[1]=sigmas[0]
        new_E[:][0]=E[:][1]
        new_E[:][1]=E[:][0]

    if abs(new_sigmas[0])<abs(new_sigmas[2]):
        new_sigmas[0]=sigmas[2]
        new_sigmas[2]=sigmas[0]
        new_E[:][0]=E[:][2]
        new_E[:][2]=E[:][0]

    if abs(new_sigmas[1])<abs(new_sigmas[2]):
        new_sigmas[1]=sigmas[2]
        new_sigmas[2]=sigmas[1]
        new_E[:][1]=E[:][2]
        new_E[:][2]=E[:][1]

    return new_sigmas, new_E

def my_p3p(points3D, points2D, cameraMatrix, distCoeffs):

    # preprocessing
    x = deepcopy(points3D)
    y = deepcopy(points2D)
    y = np.concatenate((y, np.ones((3, 1))), axis=1)

    x1, x2, x3 = x
    y1, y2, y3 = y

    # normalize y
    y1 = y1/np.linalg.norm(y1, 2)
    y2 = y2/np.linalg.norm(y2, 2)
    y3 = y3/np.linalg.norm(y3, 2)

    # compute a_ij, b_ij
    b12 = -2.0*np.dot(y1, y2)
    b13 = -2.0*np.dot(y1, y3)
    b23 = -2.0*np.dot(y2, y3)

    d12 = x1-x2
    d13 = x1-x3
    d23 = x2-x3

    a12 = np.linalg.norm(d12, 2)**2
    a13 = np.linalg.norm(d13, 2)**2
    a23 = np.linalg.norm(d23, 2)**2

    # get D1,D2, and find r
    c31 = -0.5*b13
    c23 = -0.5*b23
    c12 = -0.5*b12
    blob = (c12*c23*c31-1.0)

    s31_squared = 1.0 - c31*c31
    s23_squared = 1.0 - c23*c23
    s12_squared = 1.0 - c12*c12

    p3 = (a13*(a23*s31_squared - a13*s23_squared))
    p2 = 2.0*blob*a23*a13 + a13*(2.0*a12 + a13) * s23_squared + a23*(a23 - a12)*s31_squared
    p1 = a23*(a13 - a23)*s12_squared - a12*a12*s23_squared - 2.0*a12*(blob*a23 + a13*s23_squared)
    p0 = a12*(a12*s23_squared - a23*s12_squared)

    roots = np.roots([p3, p2, p1, p0])
    for r in roots[::-1]:
        if np.isreal(r) == True:
            root = np.real(r)
            break
    # solve eigenvalue [ E, sigma_1, sigma_2]

    A00 = a23*(1.0 - root)
    A01 = (a23*b12)*0.5
    A02 = (a23*b13*root)*(-0.5)
    A11 = a23 - a12 + a13*root
    A12 = b23*(a13*root - a12)*0.5
    A22 = root*(a13 - a23) - a12

    A=np.array([[A00,A01,A02],[A01,A11,A12],[A02,A12,A22]])
    sigmas, E = np.linalg.eig(A)
    new_sigmas, new_E=order_eigenvalue(sigmas,E)
    sigma = max(0,-1*new_sigmas[1]/new_sigmas[0])

    # s = +- (-sigma_2/sigma_1)**(1/2)
    s = np.zeros((2))
    s[0] = np.sqrt(sigma)
    s[1] = -1*np.sqrt(sigma)

    # find valid tau
    good_lamb = []

    for i in range(2):
        avaliable_root = []
        w0 = (new_E[1][0]-s[i]*new_E[1][1])/(s[i]*new_E[0][1]-new_E[0][0])
        w1 = (new_E[2][0]-s[i]*new_E[2][1])/(s[i]*new_E[0][1]-new_E[0][0])

        aa = (a13-a12)*w1*w1-a12*b13*w1-a12
        bb = a13*b12*w1-a12*b13*w0-2*w0*w1*(a12-a13)
        cc = (a13-a12)*w0*w0+a13*b12*w0+a13

        two_roots = np.roots([aa, bb, cc])
        for r in two_roots:
            if np.isreal(r) == True:
                if r >= 0:
                    avaliable_root.append(np.real(r))

        tau = np.array(avaliable_root)

    # compute the t_k > 0 and get A_k
        for t in tau:
            lamb2 = np.sqrt(a23/(t*(b23+t)+1.0))
            lamb3 = lamb2*t
            lamb1 = w0*lamb2+w1*lamb3

            if lamb1 >= 0:
                good_lamb.append([lamb1, lamb2, lamb3])

    # compute X_inv
    X=mix(d12, d13)
    X_inv = np.linalg.inv(X)

    answer = []
    # for each valid A, calculate R, T
    for lambs in good_lamb:

        # compute Y_k
        z1 = lambs[0]*y1-lambs[1]*y2
        z2 = lambs[0]*y1-lambs[2]*y3
        Y = mix(z1, z2)

        # compute R_k
        R = Y@X_inv

        # compute T_k
        T = lambs[0]*y1-R@x1
        T=T.reshape((3,1))

        answer.append([R,T])

    return answer

def calculate_distance(y,y_pred,t):
    s=(y[0]-y_pred[0])**2 + (y[1]-y_pred[1])**2
    if s <t**2:
        return 1
    else:
        return 0

def ransac(R,T,points3D, points2D, num,K,p):
    cnt=0
    l=[]
    for i in range(num):
        x=np.array(points3D[i]).reshape(3,1)
        y=points2D[i]
        tmp=R@x+T
        tmp/=tmp[-1]
        inlier=calculate_distance(y,tmp,100)
        if inlier==1:
            l.append(i)
        cnt+=inlier
    return cnt

def my_pnp(points3D, points2D, cameraMatrix, distCoeffs):
    num_points=points3D.shape[0]
    times=100
    best_inl=0
    for i in range(times):
        num_list=random.sample(range(num_points),3)
        R_T=my_p3p(points3D[num_list], points2D[num_list], cameraMatrix, distCoeffs)
        for ans in R_T:
            R,T =ans
            inl=ransac(R,T,points3D, points2D, num_points,cameraMatrix,num_list)
            if inl>best_inl:
                best_inl=inl
                best_list=num_list
                best_ans=ans
   
    print(f"best inliers:{best_inl}")

    return best_ans



def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])


    result = my_pnp(points3D, points2D, cameraMatrix, distCoeffs)
    q=R.from_matrix(result[0]).as_rotvec()
    t=result[1]

    # opencv_ans=cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
    # q=opencv_ans[1]
    # t=opencv_ans[2]

    return q,t


def find_camera_position(rvec, tvec):
    r_matrix=R.from_rotvec(rvec.reshape(1,3)).as_matrix().reshape(3,3)
    t_matrix = tvec.reshape(3,1)
    R_T=np.concatenate((r_matrix, t_matrix), axis=1)
    tmp=np.array([[0,0,0,1]])
    R_T=np.concatenate((R_T,tmp),axis=0)
    R_inverse=np.linalg.inv(R_T)
    R_matrix=R_inverse[:3,:3]
    T_matrix=R_inverse[:3,3]
    return R_matrix,T_matrix

def differences(rotq,tvec,rotq_gt,tvec_gt):
    d_t=np.linalg.norm(tvec-tvec_gt,2)
    nor_rotq=rotq/np.linalg.norm(rotq)
    nor_rotq_gt=rotq_gt/np.linalg.norm(rotq_gt)
    dif_r=np.clip(np.sum(nor_rotq*nor_rotq_gt),0,1)
    d_r=np.degrees(np.arccos(2*dif_r*dif_r-1))

    return d_r, d_t


print("processing ...")

# Process model descriptors
desc_df = average_desc(train_df, points3D_df)
kp_model = np.array(desc_df["XYZ"].to_list())
desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
df=pd.DataFrame(columns=['rotation','position'])
differences_rotation=[]
differences_transition=[]

for i in trange(164,293): #293
    # Load quaery image
    idx = i
    fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
    rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

    # Load query keypoints and descriptors
    points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
    kp_query = np.array(points["XY"].to_list())
    desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
    
    # Find correspondance and solve pnp
    rvec, tvec = pnpsolver((kp_query, desc_query),(kp_model, desc_model))

    rotation,position=find_camera_position(rvec, tvec)
    df=df.append({'rotation':rotation, 'position':position},ignore_index=True)

    rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()
    tvec = tvec.reshape(1,3)


    # Get camera pose groudtruth 
    ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
    rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
    tvec_gt = ground_truth[["TX","TY","TZ"]].values

    d_r,d_t=differences(rotq,tvec,rotq_gt,tvec_gt)

    print(d_r)
    print(d_t)

    differences_rotation.append(d_r)
    differences_transition.append(d_t)

differences_rotation=np.array(differences_rotation)
differences_transition=np.array(differences_transition)
err_r=np.median(differences_rotation)
err_t=np.median(differences_transition)

print(f"pose error:{err_t}, rotation error:{err_r}")

df.to_pickle("./camera_position.pkl")

