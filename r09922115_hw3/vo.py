import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import matplotlib.pyplot as plt
import ipdb

os.chdir("/Users/yun/Desktop/3d_cv/hw/hw3/homework3-Chushihyun-master") 

def draw_match(img1,img2,points1,points2):
    num=points1.shape[0]

    keypoints1=[]
    keypoints2=[]
    matches=[]
    for point in points1:
        keypoints1.append(cv.KeyPoint(point[0],point[1],1))
    for point in points2:
        keypoints2.append(cv.KeyPoint(point[0],point[1],1))
    for i in range(num):
        matches.append(cv.DMatch(i,i,1))

    img_draw_match = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_DEFAULT)
    cv.namedWindow('match',0)
    cv.resizeWindow('match', 1000, 400)
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return 0


def draw_camera(position, rotation):
    p = position
    r = rotation
    model = o3d.geometry.LineSet()
    model.points = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
    model.lines = o3d.utility.Vector2iVector(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
    color = np.array([1, 0, 0])
    model.colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))
    model.scale(0.5, np.zeros(3))
    model.rotate(r)
    model.translate(p)
    return model

def sort_inliers(points1,points2,inlier):
    new_points1=[]
    new_points2=[]
    for i in range(len(inlier)):
        if inlier[i][0]!=0:
            new_points1.append(points1[i])
            new_points2.append(points2[i])
    return np.array(new_points1),np.array(new_points2)


def sort_inliers_3d(points1,points2,points3d,inlier):
    new_points1=[]
    new_points2=[]
    new_points3d=[]
    for i in range(len(inlier)):
        if inlier[i][0]!=0:
            new_points1.append(points1[i])
            new_points2.append(points2[i])
            new_points3d.append(points3d[:,i])

    for i in range(len(new_points3d)):
        point=new_points3d[i]
        point=point/point[3]
        point=point[:3]
        new_points3d[i]=point

    return np.array(new_points1),np.array(new_points2),np.array(new_points3d)


def rescale_ratio(points2_prev,points3d_prev,points1,points3d):
    same_points=[]
    for i in range(points2_prev.shape[0]):
        for j in range(points1.shape[0]):
            if np.all(points2_prev[i]==points1[j]):
                same_points.append([i,j])

    # print(len(same_points))
    if len(same_points)<=1:
        return 1

    # find pairs
    sample_times=13
    sample_times=min(sample_times,len(same_points))
    ratios=[]
    for t in range(sample_times):
        index1,index2=np.random.choice(len(same_points), 2, replace=False)
        idx1_prev=same_points[index1][0]
        idx1=same_points[index1][1]
        idx2_prev=same_points[index2][0]
        idx2=same_points[index2][1]

        norm=np.linalg.norm(points3d[idx1]-points3d[idx2])
        norm_prev=np.linalg.norm(points3d_prev[idx1_prev]-points3d_prev[idx2_prev])
        ratio=norm/(norm_prev+0.000001)
        ratios.append(ratio)

    scale=np.median(np.array(ratios))

    return scale

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        queue = mp.Queue()

        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        Rt_prev=np.eye(4, dtype=np.float64)
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    
                    # build Rt
                    Rt = np.concatenate([R, t], -1)
                    Rt = np.concatenate([Rt, np.zeros((1, 4))], 0)
                    Rt[-1, -1] = 1.

                    # upate Rt_prev
                    Rt_prev=Rt_prev@Rt

                    R=Rt_prev[:3,:3]
                    t=Rt_prev[:3,3]
                    
                    model=draw_camera(t,R)
                    vis.add_geometry(model)

                    pass
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def get_correspond_points(self,img1,img2):
        
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)

        points1 = np.array([kp1[m.queryIdx].pt for m in matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in matches])

        return points1,points2

    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        frame_path_prev=self.frame_paths[0]
        for frame_path in self.frame_paths[1:]:

            img1 = cv.imread(frame_path_prev)
            img2 = cv.imread(frame_path)

            points1,points2= self.get_correspond_points(img1,img2)
            # draw_match(img1,img2,points1,points2)

            essential,inlier=cv.findEssentialMat(points1,points2,self.K)
            points1,points2=sort_inliers(points1,points2,inlier)
            # draw_match(img1,img2,points1,points2)

            val,R,t,inlier,traingularpoints=cv.recoverPose(essential,points1,points2,self.K,distanceThresh=400)
            points1,points2,points3d=sort_inliers_3d(points1,points2,traingularpoints,inlier)
            # draw_match(img1,img2,points1,points2)

            # rescale t
            if frame_path_prev==self.frame_paths[0]:
                # first time
                ratio=1
            else:
                ratio=rescale_ratio(points2_prev,points3d_prev,points1,points3d)
                ratio*=ratio_prev
                ratio=min(max(ratio,0.5),2)
            t*=ratio

            queue.put((R, t))
            
            ratio_prev=ratio
            points2_prev=points2
            points3d_prev=points3d
            frame_path_prev=frame_path
            
            cv.imshow('frame', img1)
            if cv.waitKey(30) == 27: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
