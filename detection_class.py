# from sklearn.cluster import KMeans
import numpy as np 
import ros_numpy
import rospy, cv2,torch
import pdb
import colorsys
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import PointCloud2,Image
# from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker,MarkerArray

wlh_list = {
    'person': torch.tensor([0.67, 0.73, 1.77]),
    'bicycle':torch.tensor([0.60, 1.70, 1.28]),
    'car': torch.tensor([1.95, 4.62, 1.73]),
    'motorcycle': torch.tensor([0.77, 2.11, 1.47]),
    'bus': torch.tensor([2.93, 11.23, 3.47]),
    'truck': torch.tensor([2.51, 6.93, 2.84]),
}
names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
         9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
         16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
         25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 
         33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
         40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
         49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
         58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
         66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
         74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

TRANSg = torch.tensor([[-0.9999, 0.0093,-0.0023, 0.0518159820647960282],
                  [ 0.0015,-0.0824,-0.9966, 0.17792806925426059], 
                  [-0.0094,-0.9966,-0.0824, 0.1966839105194658], 
                  [0     ,0      ,0      ,1      ]],device='cuda',dtype=torch.float64)

R = TRANSg.T[:3,:3]
T = TRANSg.T[-1,:3]

intrin_mg = torch.tensor([[604.7271728515625,0.0   ,320.4165344238281],
                     [0.0   ,603.350341796875,232.33656311035156],
                     [0.0   ,0.0   ,1.0   ]],device='cuda',dtype=torch.float64)

intrin_m = np.array([[604.7271728515625,0.0   ,320.4165344238281],
                     [0.0   ,603.350341796875,232.33656311035156],
                     [0.0   ,0.0   ,1.0   ]])

TRANS = np.array([[-0.9999, 0.0093,-0.0023, 0.0518159820647960282],
                  [ 0.0015,-0.0824,-0.9966, 0.17792806925426059], 
                  [-0.0094,-0.9966,-0.0824, 0.1966839105194658], 
                  [0     ,0      ,0      ,1      ]])

cls_idx_trans = {0:4,1:0,2:2,3:3,5:6,7:6}

def get_center(pcs, mode):
    center = np.zeros(4)
    if mode == 1:
        return np.mean(pcs,axis=0)
    elif mode == 2:
        # detect_cluster = KMeans(n_clusters=1).fit(pcs)
        # return detect_cluster.cluster_centers_
        return 0
    elif mode == 3:
        mean = np.mean(pcs,axis=0)
        x = np.mean(pcs[:,0])
        z = np.mean(pcs[:,2])
        delete_num = int(len(pcs)*0.5)
        y = np.mean(np.delete(pcs[:,1],np.argpartition(-pcs[:,1],-delete_num)[-delete_num:]))
        print(f'mean_points:{mean}, x:{x}, z:{z}, y:{y}')
        return np.array([x,y,z,1])
    elif mode == 4:
        pcs = pcs[pcs[:,2] > -0.95] # delete the ground points,preventing it from caculating mean points
        # print(f'zmean: {np.min(pcs[:,2])}')
        filter_nums = int(len(pcs)*0.5)
        # pdb.set_trace()
        _,idx = torch.topk(torch.sum(pcs**2,dim=1),filter_nums,largest=False,sorted=False)
        filter_points = pcs[idx]
        mean_point = torch.mean(filter_points,dim=0)
        # return np.concatenate((mean_point,[1]))
        return mean_point
    else:
        raise Exception('incorrect mode!')


class detect_result:
    def __init__(self,
                 pointclouds: torch.tensor, 
                 bbox,
                 obj_class: int,
                 score,
                 Trans):

        self.pointclouds, self.obj_class, self.points_num, self.name = pointclouds, obj_class, len(pointclouds), names[obj_class]
        self.score, self.bbox, self.wlh, self.center = score, bbox, None, None
        self.wlh = wlh_list[self.name] if self.obj_class < 8 and self.obj_class not in [4,6] else [0, 0, 0]
        self.center = get_center(self.pointclouds, 4)
        self.Trans = Trans       

    # def kmeans(self):
    #     detect_cluster = KMeans(n_clusters=1).fit(self.pointclouds)
    #     # detect_pic_cluster = k_means(n_clusters=1).fit()
    #     self.center = detect_cluster.cluster_centers_
    #     # pass

    def get_result(self):
        detect_cls = torch.tensor([cls_idx_trans[self.obj_class]],device='cuda')
        score = torch.tensor([self.score],device='cuda')
        vandry = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0],device='cuda')
        P_GLO = torch.matmul(self.Trans,self.center) #transfer to world coordinate
        # pdb.set_trace()
        result = torch.cat((P_GLO[:-1],self.wlh.to('cuda'),vandry,score,detect_cls),dim=0)
        # pdb.set_trace()
        return result
    
    # def get_result_list(self):
    #     result = {
    #         "sample_token": '0',
    #         "translation": self.center,
    #         "size": self.wlh,
    #         "rotation": [1.0, 0.0, 0.0, 0.0],
    #         "velocity": [0.0, 0.0],
    #         "detection_name": self.name,
    #         "detection_score": self.score,
    #         "attribute_name": 'xxx'
    #     }
    #     return result

    def get_bottom_corner(self):
        x,y = self.center[0],self.center[1]
        l,h =self.wlh[1],self.wlh[2]
        return np.array([[x-l/2,y-h/2],[x-l/2,y+h/2],[x+l/2,y-h/2],[x+l/2,y+h/2]])

    def __getitem__(self,item):
        self.result = self.get_result()
        # self.bottom_corner = self.get_bottom_corner()
        return self.result

    def __repr__(self) -> str:
        return f'''class:{names[self.obj_class]}
                 score:{self.score}
                 wlh:{self.wlh}
                 center:{self.center}
                 points_num:{self.points_num}'''
    

def project_points_to_image(pcs_xyz):
    cam_pcs2 = torch.matmul(TRANSg, pcs_xyz.T).T[:, :3]
    cam_pcs = torch.matmul(intrin_mg, cam_pcs2.T).T 
    nom = cam_pcs / cam_pcs[:, 2, None].repeat(1,3)
    return nom

def msg_to_array(msg):
    # pdb.set_trace()
    pc_array = ros_numpy.numpify(msg)
    # pdb.set_trace()
    pc = np.zeros([len(pc_array),4])
    pc[:,0] = pc_array['x']
    pc[:,1] = pc_array['y']
    pc[:,2] = pc_array['z']
    pc[:,3] = 1
    pc_left = np.zeros([len(pc_array),3])
    pc_left[:,0] = pc_array['intensity']
    pc_left[:,1] = pc_array['timestamp']
    pc_left[:,2] = pc_array['ring']
    return pc, pc_left

def pointcloud_to_image(subscriber_lidar):
    pcs_xyz, pcs_lf = msg_to_array(subscriber_lidar)
    # pdb.set_trace()  
    # cam_pcs2 = np.dot(TRANS, pcs_xyz.T).T[:, :3] # transfer from lidar coordinate to camera coordinate  
    cam_pcs2 = np.dot(pcs_xyz[:,:3],R) + T
    btx = cam_pcs2[cam_pcs2[:,-1]>0] # filter z < 0 points
    cam_pcs = np.dot(intrin_m, btx.T).T # transfer to pixel coordinate
    # nom = np.array([m/m[2] for m in cam_pcs])
    nom = cam_pcs / cam_pcs[:, 2, None].repeat(3, -1)
    cam_pix2 = nom[(nom[:,0] < 639) & (nom[:,1] < 479) & (nom[:,0] > 1) & (nom[:,1] > 1)][:,:2].astype('int16') #filer 0<x<640 0<y<480
    # only for viz
    mask = (nom[:,0] < 639) & (nom[:,1] < 479) & (nom[:,0] > 1) & (nom[:,1] > 1)
    valid_pc_idx = np.arange(len(pcs_xyz))[cam_pcs2[:,-1] > 0][mask]
    # pdb.set_trace()
    other_pc_idx = np.concatenate((np.arange(len(pcs_xyz))[cam_pcs2[:,-1] <= 0], np.arange(len(pcs_xyz))[cam_pcs2[:,-1] > 0][~mask]))
    pcs_in_camera = pcs_xyz[valid_pc_idx]
    cam_pcs_lf = pcs_lf[valid_pc_idx]
    other_pcs = {'location': pcs_xyz[other_pc_idx], 'lf': pcs_lf[other_pc_idx]}
    # pdb.set_trace()
    return pcs_in_camera, cam_pix2,cam_pcs_lf,subscriber_lidar.header.stamp, other_pcs

def pointcloud_to_image_g(subscriber_lidar):
    pcs_xyz, pcs_lf = msg_to_array(subscriber_lidar)
    # pdb.set_trace()  
    # cam_pcs2 = np.dot(TRANS, pcs_xyz.T).T[:, :3] # transfer from lidar coordinate to camera coordinate 
    pcs_xyz = torch.tensor(pcs_xyz).to('cuda')
    pcs_lf = torch.tensor(pcs_lf).to('cuda') 
    cam_pcs2 = torch.matmul(pcs_xyz[:,:3],R) + T
    btx = cam_pcs2[cam_pcs2[:,-1]>0] # filter z < 0 points
    cam_pcs = torch.matmul(intrin_mg, btx.T).T # transfer to pixel coordinate
    # nom = np.array([m/m[2] for m in cam_pcs])
    nom = cam_pcs / cam_pcs[:, 2, None].repeat(1,3)   

    mask = (nom[:,0] < 639) & (nom[:,1] < 479) & (nom[:,0] > 1) & (nom[:,1] > 1)
    cam_pix2 = nom[mask][:,:2].to(torch.int16) #filer 0<x<640 0<y<480
    valid_pc_idx = torch.arange(len(pcs_xyz),device='cuda')[cam_pcs2[:,-1] > 0][mask]
    # pdb.set_trace()
    other_pc_idx = torch.cat((torch.arange(len(pcs_xyz),device='cuda')[cam_pcs2[:,-1] <= 0], torch.arange(len(pcs_xyz),device='cuda')[cam_pcs2[:,-1] > 0][~mask]))
    pcs_in_camera = pcs_xyz[valid_pc_idx]
    # pdb.set_trace()
    cam_pcs_lf = pcs_lf[valid_pc_idx]
    # pdb.set_trace()
    other_pcs = {'location': pcs_xyz[other_pc_idx].to('cpu').numpy(), 'lf': pcs_lf[other_pc_idx].to('cpu').numpy()}
    # pdb.set_trace()
    return pcs_in_camera, cam_pix2,cam_pcs_lf,subscriber_lidar.header.stamp, other_pcs


def init_marker(position, id):
    marker = Marker()
    marker.type = Marker.SPHERE
    # marker.actiom = Marker.ADD
    marker.header.frame_id = "PandarXT-32"
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    size = 0.2
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration(0.1)
    # marker.header.stamp = rospy.Time.now()
    marker.ns = 'spheres'
    marker.id = id
    return marker

def init_bbox_marker(center,rotation,size,color,id):
    marker = Marker()
    # 设置矩形框的坐标系，时间戳，命名空间和ID
    marker.ns = "rectangle"
    marker.id = id
    marker.header.frame_id = "PandarXT-32"
    # 设置矩形框的类型，动作，位置，姿态，尺寸，颜色和持续时间
    marker.type = Marker.CUBE
    # marker.action = Marker.ADD
    marker.pose.position.x = center[0] # 矩形框的中心点的x坐标
    marker.pose.position.y = center[1] # 矩形框的中心点的y坐标
    marker.pose.position.z = center[2] # 矩形框的中心点的z坐标
    marker.pose.orientation.x = rotation[1] # 矩形框的旋转四元数的x分量
    marker.pose.orientation.y = rotation[2] # 矩形框的旋转四元数的y分量
    marker.pose.orientation.z = rotation[3] # 矩形框的旋转四元数的z分量
    marker.pose.orientation.w = rotation[0] # 矩形框的旋转四元数的w分量
    marker.scale.x = size[0] # 矩形框的长度
    marker.scale.y = size[1] # 矩形框的宽度
    marker.scale.z = size[2] # 矩形框的高度
    marker.color.r = color[0] # 矩形框的颜色的红色分量
    marker.color.g = color[1] # 矩形框的颜色的绿色分量
    marker.color.b = color[2] # 矩形框的颜色的蓝色分量
    marker.color.a = 1 # 矩形框的颜色的透明度
    marker.lifetime = rospy.Duration(0.1) 
    return marker

def init_markerarray(info, type):
    markerarray = MarkerArray()
    id = 1
    if type == 'sphere':
        for i in info: # info:[n,3],xyz
            marker = init_marker(i, id)
            markerarray.markers.append(marker)
            id += 1
    elif type == 'bbox': #info,dict,x,y,z,w,l,h,rotation
         for result in info:
            marker = init_bbox_marker(result['center'],result['wxyz'],result['size'],id)
            markerarray.markers.append(marker)
            id += 1
    # markerarray.header.stamp = rospy.Time.now()
    return markerarray

def assign_colors_to_tracking_ids(tracking_ids):
    """
    assign unique color for each tracklets
    :param tracking_ids: list, all tracking id at the current frame
    :return: unique colors of each tracklets
    """
    # pdb.set_trace()
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(tracking_ids / 20, 1.0, 1.0))

def draw_box3d_image(image, qs, tra_id, img_size=(480, 640), color=(255,255,255), thickness=4):
	''' Draw 3d bounding box in image
	    qs: (8,2) array of vertices for the 3d box in following order:
	        1 -------- 0
	       /|         /|
	      2 -------- 3 .
	      | |        | |
	      . 5 -------- 4
	      |/         |/
	      6 -------- 7
	'''

	def check_outside_image(x, y, height, width):
		if x < 0 or x >= width: return True
		if y < 0 or y >= height: return True

	# if 6 points of the box are outside the image, then do not draw
    
    # color = assign_colors_to_tracking_ids(tra_id)
	pts_outside, color, tra_info = 0, assign_colors_to_tracking_ids(tra_id), 'ID = ' + str(tra_id)
	for index in range(8):
		check = check_outside_image(qs[index, 0], qs[index, 1], img_size[0], img_size[1])
		if check: pts_outside += 1
	if pts_outside >= 6: return image, False

	# actually draw
	if qs is not None:
		qs = qs.astype(np.int32)
		for k in range(0,4):
			i,j=k,(k+1)%4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

			i,j=k+4,(k+1)%4 + 4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

			i,j=k,k+4
			image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
        
    
    #   cv2.putText(image, tra_info, (qs[1, 0], qs[1, 1] - 5), cv2.FONT_HERSHEY_PLAIN, 3.0, color, 4)
	return image, True
