#! /usr/bin/python3

import pdb
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    cv2.imshow("s", cv_image)
    cv2.waitKey(1)
    cv2.imwrite("/home/agilex/test.png", cv_image)
    rospy.loginfo("Hello")




if __name__ == '__main__':
    rospy.init_node('image', anonymous=True)

    rospy.Subscriber("/camera/color/image_raw", Image, callback)

    rospy.spin()
