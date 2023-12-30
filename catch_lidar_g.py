#! /home/agilex/anaconda3/envs/yolov5.1/lib/python3.8
import rospy
import ros_numpy
import pdb
import time
import cv2
import message_filters
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import PointCloud2,Image
from sensor_msgs import point_cloud2
from collections import namedtuple
import torch
# from line_profiler import LineProfiler

# TRANS = np.array([[-0.9844,-0.1761,0.0009,-0.2492],
#                   [-0.0322,-0.1752,-0.9840,0.3840], 
#                   [0.1731,-0.9687,-0.1781,-0.0883], 
#                   [0     ,0      ,0      ,1      ]])

TRANS = torch.tensor([[-0.9999, 0.0093,-0.0023,-0.0018159820647960282],
                  [ 0.0015,-0.0824,-0.9966, 0.07792806925426059], 
                  [-0.0094,-0.9966,-0.0824, 0.1966839105194658], 
                  [0     ,0      ,0      ,1      ]],device='cuda',dtype=torch.float64)

TRANS_T = TRANS.T

# R = torch.tensor(TRANS_T[:3, :3]).to('cuda')
# T = torch.tensor(TRANS_T[:3, -1]).to('cuda')
R = TRANS_T[:3,:3]
T = TRANS_T[:3,-1]

intrin_m = torch.tensor([[604.7271728515625,0.0   ,320.4165344238281],
                        [0.0   ,603.350341796875,232.33656311035156],
                        [0.0   ,0.0   ,1.0   ]],device='cuda:0',dtype=torch.float64)

def msg_to_array(msg):
    pc_array = ros_numpy.numpify(msg)
    pc = np.zeros([len(pc_array),4])
    pc[:,0] = pc_array['x']
    pc[:,1] = pc_array['y']
    pc[:,2] = pc_array['z']
    pc[:,3] = 1
    return pc, pc_array['intensity']


# @profile
def multi_callback(subscriber_lidar,subscriber_camera):
    # points = point_cloud2.read_points_list(subscriber_lidar, field_names=("x", "y", "z"))
    # pcs_xyz = np.array([[pc.x, pc.y, pc.z, 1] for pc in points]) #transfer the message type to numpy arrar
    # project the pointcloud onto the image
    pc_array = ros_numpy.numpify(subscriber_lidar)
    pcs_xyz = np.zeros([len(pc_array),3])
    pcs_xyz[:,0] = pc_array['x']
    pcs_xyz[:,1] = pc_array['y']
    pcs_xyz[:,2] = pc_array['z']
    # pcs_xyz[:,3] = 1
    pcs_lf =  pc_array['intensity']
    # pcs_xyz, pcs_lf = msg_to_array(subscriber_lidar)
    pcs_xyz = torch.tensor(pcs_xyz).to('cuda')
    # pdb.set_trace()
    # cam_pcs2 = np.dot(pcs_xyz,TRANS_T)[:, :3] # transfer from lidar coordinate to camera coordinate
    cam_pcs2 = torch.matmul(pcs_xyz[:,:3],R) + T
    
    # cam_pcs2 = torch.matmul(cam_pcs,m1.T)  # degrade the matrix from 4 dimensions to 3 dimensions
    btx = cam_pcs2[cam_pcs2[:,-1]>0] # filter z < 0 points
    # pdb.set_trace()
    cam_pcs = torch.matmul(intrin_m, btx.t()).t() # transfer to pixel coordinate
    # nom = np.array([m/m[2] for m in cam_pcs])
    
    nom = cam_pcs / cam_pcs[:, 2, None].repeat(1,3)
    pdb.set_trace()
    # nom = cam_pcs / cam_pcs[:, 2, None].repeat(3, -1)
    cam_pix2 = nom[(nom[:,0] < 639) & (nom[:,1] < 479) & (nom[:,0] > 1) & (nom[:,1] > 1)][:,:2].to(torch.int16) #filer 0<x<640 0<y<480
    # only for viz
    mask = (nom[:,0] < 639) & (nom[:,1] < 479) & (nom[:,0] > 1) & (nom[:,1] > 1)
    valid_pc_idx = torch.arange(len(pcs_xyz))[cam_pcs2[:,-1] > 0][mask]
    high_int_mask = pcs_lf[valid_pc_idx] >= 127
    

    other_pc_idx = torch.cat((torch.arange(len(pcs_xyz))[cam_pcs2[:,-1] <= 0], torch.arange(len(pcs_xyz))[cam_pcs2[:,-1] > 0][~mask]))
    pcs_in_camera = pcs_xyz[valid_pc_idx]
    cam_pcs_lf = pcs_lf[valid_pc_idx]
    # other_pcs = {'location': pcs_xyz[other_pc_idx], 'lf': pcs_lf[other_pc_idx]}
    location = pcs_xyz[other_pc_idx]
    lf = pcs_lf[other_pc_idx]
    # cam_pix2 = np.dot(cam_pix, m2.T)
    # cam_pix2 = cam_pix[:,:2]
    # # pdb.set_trace()
    # # cam_pix2 = np.rint(cam_pix2)
    # cam_pix2 = cam_pix2.astype('int16')
    #get camera info
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(subscriber_camera, desired_encoding='bgr8')
    # pdb.set_trace() 
    cam_pix2 = cam_pix2.to('cpu').numpy()
    cv_image[cam_pix2[:, 1], cam_pix2[:, 0], :] = 255
    cv_image[cam_pix2[high_int_mask, 1], cam_pix2[high_int_mask, 0], :] = 0 
    cv2.imshow("s", cv_image)
    cv2.waitKey(1)
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s ,%s", subscriber_camera.data, subscriber_lidar.data) 



    # pdb.set_trace()




if __name__ == '__main__':
    rospy.init_node('lidar_listener', anonymous=True)
    # rospy.Subscriber('/hesai/pandar', PointCloud2, callback)
    # rospy.Subscriber("/camera/color/image_raw", Image, callback)
    subscriber_lidar = message_filters.Subscriber('/hesai/pandar',PointCloud2)
    subscriber_camera = message_filters.Subscriber("/camera/color/image_raw",Image)
  
    sync = message_filters.ApproximateTimeSynchronizer([subscriber_lidar,subscriber_camera],10,0.1)

    sync.registerCallback(multi_callback)
    rospy.spin()
