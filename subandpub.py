import ros_numpy
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters
# from detection_class import detet_result
from sensor_msgs.msg import PointCloud2,Image
from sensor_msgs import point_cloud2
import pdb
from visualization_msgs.msg import Marker,MarkerArray
global counttime
counttime = 0

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

def init_marker(position,i):
    marker = Marker()
    marker.type = Marker.SPHERE
    # marker.actiom = Marker.ADD
    marker.header.frame_id = "map"
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    size = 2
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.lifetime = rospy.Duration()
    # marker.header.stamp = rospy.Time.now()
    marker.ns = "spheres"
    marker.id = i
    return marker

def init_markerarray(positions):
    markerarray = MarkerArray()
    id = 1
    for i in positions:
        marker = init_marker(i, id)
        markerarray.markers.append(marker)
        id += 1
    # markerarray.header.stamp = rospy.Time.now()
    return markerarray


def multi_callback(subscriber_lidar,subscriber_camera):
    bridge = CvBridge()
    # pdb.set_trace()
    cv_image = bridge.imgmsg_to_cv2(subscriber_camera, desired_encoding='bgr8')
    cv2.imshow("s",cv_image)
    cv2.waitKey(1)
    global counttime
    counttime += 1
    print(f'runtimes:',counttime)
    pcs,pcs_left = msg_to_array(subscriber_lidar)
    mask = np.arange(len(pcs))[pcs[:,1]<0]
    pcs1 = pcs[mask][:,:3]
    pcs_lf = pcs_left[mask]
    # pdb.set_trace()
    dt = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'timestamp', 'ring'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<f8', '<u2'],
    'offsets': [0, 4, 8, 16, 24, 32],
    'itemsize': 48
    })
    data = np.zeros(len(pcs1),dtype=dt)
    data['x'] = pcs1[:,0]
    data['y'] = pcs1[:,1]
    data['z'] = pcs1[:,2]
    data['intensity'] = pcs_lf[:,0]
    data['timestamp'] = pcs_lf[:,1]
    data['ring'] = pcs_lf[:,2]
    pcs_msg = ros_numpy.msgify(PointCloud2,data)
    pcs_msg.header.frame_id = "map"

    # pc_array = ros_numpy.numpify(subscriber_lidar)
    # pdb.set_trace()
    # pcs1 = pc_array
    # # pdb.set_trace()
    # pcs_msg = ros_numpy.msgify(PointCloud2,pcs1)
    puber.publish(pcs_msg)
    center = np.mean(pcs1,axis=0)
    c2 = center+5
    c3 = center-5
    center = np.concatenate([center,c3,c2],axis=0).reshape(-1,3)
    # sphere_msg = init_marker(center)
    sphere_msg = init_markerarray(center)
    # pdb.set_trace()
    pubmarker.publish(sphere_msg)


    



if __name__ == '__main__':
    rospy.init_node('sandp', anonymous=True)

    puber = rospy.Publisher("fil_pcs",PointCloud2,queue_size=10)
    pubmarker = rospy.Publisher("sphere_mark",MarkerArray,queue_size=10)
    subscriber_lidar = message_filters.Subscriber('/hesai/pandar',PointCloud2)
    subscriber_camera = message_filters.Subscriber("/camera/color/image_raw",Image)
  
    sync = message_filters.ApproximateTimeSynchronizer([subscriber_lidar,subscriber_camera],10,0.1)

    sync.registerCallback(multi_callback)
    rospy.spin()