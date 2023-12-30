# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
# change CTRV2
from sklearn.cluster import KMeans
import rospy
import ros_numpy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

import message_filters
from detection_class2 import detect_result,project_points_to_image, pointcloud_to_image,init_bbox_marker,init_markerarray,draw_box3d_image,pointcloud_to_image_g
from sensor_msgs.msg import PointCloud2,Image
from sensor_msgs import point_cloud2
from collections import namedtuple
from visualization_msgs.msg import Marker,MarkerArray
import numpy as np
import pdb

import sys
from polytest import yolo_tracker

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadScreenshots, LoadStreams, LoadImages, myLoadImages
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

from dataloader.nusc_loader import NuScenesloader
from tracking.nusc_tracker import Tracker
import yaml
import tf
from pyquaternion import Quaternion

datatp = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'timestamp', 'ring'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<f8', '<u2'],
    'offsets': [0, 4, 8, 16, 24, 32],
    'itemsize': 48
    })

T_N = 0
WHOLE_TIME = np.array([0,0,0,0])  

def create_pointcloud2_np(cam_pcs3d,pcs_lf,other_pcs):
    '''this function is used for creating the pointcloud msg after filtering, from numpy.array form'''
    env_data = np.zeros(len(cam_pcs3d),dtype=datatp)
    env_data['x'] = cam_pcs3d[:,0]
    env_data['y'] = cam_pcs3d[:,1]
    env_data['z'] = cam_pcs3d[:,2]
    env_data['intensity'] = pcs_lf[:,0]
    env_data['timestamp'] = pcs_lf[:,1]
    env_data['ring'] = pcs_lf[:,2]
    other_location = other_pcs['location']
    other_lf = other_pcs['lf']
    other_data = np.zeros(len(other_lf),dtype=datatp)
    other_data['x'] = other_location[:,0]
    other_data['y'] = other_location[:,1]
    other_data['z'] = other_location[:,2]
    other_data['intensity'] = other_lf[:,0]
    other_data['timestamp'] = other_lf[:,1]
    other_data['ring'] = other_lf[:,2]
    final_data = np.concatenate((other_data,env_data))
    env_msg = ros_numpy.msgify(PointCloud2,final_data)
    # env_msg = ros_numpy.msgify(PointCloud2,other_data)
    # env_msg = ros_numpy.msgify(PointCloud2,env_data)
    print(f'if other data = env_data {other_data==env_data}')
    env_msg.header.frame_id = "PandarXT-32"
    return env_msg

def viz(im0,center_list:list,tp_dict:dict):
    '''this function is used for result visualization, 
       centerlist: all target pointclouds' center,
       tp_dict: target pointclods information, transfer numpy form to PointCloud2 msg form to publish'''
    if center_list: #draw center in image and publish centermsg to rviz
        points_pixel = np.concatenate(center_list).reshape(-1,4)

        centermsgs = init_markerarray(points_pixel[:3],'sphere')
        markerpuber.publish(centermsgs)

        points_pixel = project_points_to_image(points_pixel)[:,:2].astype(int)
        # im0[points_pixel[:,1],points_pixel[:,0],:] = 255
        for x in points_pixel:
            cv2.circle(im0, x, 10, (255,0,0), -10)

    if tp_dict['tp']: #draw pointclouds in the detection frames on rviz
        tp_list = tp_dict['tp']
        tp_lf_list = tp_dict['tp_lf']
        tp = np.concatenate(tp_list)
        tp_lf = np.concatenate(tp_lf_list)
        # pdb.set_trace()
        data = np.zeros(len(tp),dtype=datatp)
        data['x'] = tp[:,0]
        data['y'] = tp[:,1]
        data['z'] = tp[:,2]
        data['intensity'] = tp_lf[:,0]
        data['timestamp'] = tp_lf[:,1]
        data['ring'] = tp_lf[:,2]
        pcs_msg = ros_numpy.msgify(PointCloud2,data)
        pcs_msg.header.frame_id = "PandarXT-32"
        puber.publish(pcs_msg)

@smart_inference_mode()
def run(
        model,
        image2,
        lidar_msg,
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.6,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    #initialize
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    img01 = image2
    dataset = myLoadImages(source, cv_img=img01, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(),Profile(),Profile(),Profile(),Profile(),Profile())
    with dt[6]:
        for path, im, im0s, _, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
            # get all points that could be projected to image 
            with dt[3]:
                cam_pcs3d, cam_points2d, pcs_lf,timestamp2, other_pcs = pointcloud_to_image_g(lidar_msg)
                assert len(cam_pcs3d) == len(cam_points2d) == len(pcs_lf),'caculate wrong!'

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            #get global translation matrix
            Trans = np.array([[0,1, 0,0],
                            [1,0, 0,0],
                            [0,0,-1,0],
                            [0,0, 0,1]])
            with dt[7]:
                try:
                    (trans, rot) = tf_listener.lookupTransform('/camera_init', '/body',rospy.Time(0))
                    q = Quaternion([rot[-1], rot[0], rot[1], rot[2]])
                    q = -q if q.axis[-1] < 0 else q
                    rot_mat = q.rotation_matrix
                    trans = np.array(trans).reshape(3,1)
                    RT = np.concatenate((rot_mat,trans),axis=1)
                    Trans = np.concatenate((RT,np.array([[0,0,0,1]])),axis=0)
                    # pdb.set_trace()
                except:
                    print('no trans')
            Trans_i = np.linalg.inv(Trans)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                center_list = []  # list of centers
                tp_dict = {'tp':[],'tp_lf':[]} #target points dict,include location list and other imformation list
                result_det_list = [] # detection result used for tracking
                target_mask = np.zeros((len(cam_points2d),),dtype=bool)
                all_box_mask = np.zeros((len(cam_points2d),),dtype=bool)
                tracking_mask = np.copy(target_mask)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    det = det.to(torch.device("cpu")).numpy()

                    boxmsg = MarkerArray()
                    id = 1
                    # Write results
                    with dt[4]:
                        for *xyxy, conf, cls in reversed(det):
                            if cls in [0,1,2,3,5,7]:
                                obj_mask = (cam_points2d[:, 0] > int(xyxy[0])) & (cam_points2d[:,1] > int(xyxy[1])) & (cam_points2d[:,0] < int(xyxy[2])) & (cam_points2d[:,1] < int(xyxy[3]))
                                bbox_index = np.arange(len(cam_points2d))[obj_mask]
                                target_mask = target_mask | obj_mask
                                bbox_points = cam_points2d[bbox_index]
                                bbox_lf = pcs_lf[bbox_index]
                                target_points = cam_pcs3d[bbox_index]
                                if target_points.size != 0: #there are some points within the detection frame
                                    detect_obj = detect_result(target_points,
                                                                    xyxy,
                                                                    int(cls),
                                                                    conf,
                                                                    Trans)   
  
                                    detect_det = detect_obj['result']
                                    result_det_list.append(detect_det)
                                    tp_dict['tp'].append(target_points)
                                    tp_dict['tp_lf'].append(bbox_lf)
                                    # pixel_list.append(bbox_points)

                                    #draw detection bbox
                                    mean_point = detect_obj.center
                                    marker = init_bbox_marker(mean_point,[0,0,0,1],detect_obj.wlh,[1,0,0],id)
                                    boxmsg.markers.append(marker)
                                    id += 1

                                    #get detection box mask
                                    box_mask = (mean_point[0]-detect_obj.wlh[0] < cam_pcs3d[:,0]) & (cam_pcs3d[:,0] < mean_point[0]+detect_obj.wlh[0]) & \
                                                (mean_point[1]-detect_obj.wlh[1]< cam_pcs3d[:,1]) & (cam_pcs3d[:,1] < mean_point[1]+detect_obj.wlh[1]) & \
                                                (-1.0 < cam_pcs3d[:,2]) & (cam_pcs3d[:,2] < detect_obj.wlh[2] - 1.0)
                                    all_box_mask = all_box_mask | box_mask
                                        
                                    center_list.append(mean_point)
                                    # pixel_list.append(np.mean(bbox_points,axis=0))
                                
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                
                # Tracking part
                with dt[5]:
                    if result_det_list:
                        det_result = np.concatenate(result_det_list).reshape(-1,14)
                        det_result = det_result[det_result[:,2] < 0.6] # limit z < 0.6

                        polyloader.update(det_result)
                        track_info = yolo_tracker(polyloader,polytracker,track_result)

                        pcs_glo = np.dot(Trans,cam_pcs3d.T).T
                        image = cv2.UMat(np.copy(img01))

                        for result in track_info:
                            # q = Quaternion(track_rotation[idx])
                            # q = -q if q.axis[-1] < 0 else q
                            # R_ = np.linalg.inv(q.rotation_matrix)
                            # bboxr = np.dot(R_,bbox)
                            # pcs_gr = np.dot(R_,pcs_glo[:,:-1].T).T
                            # box_mask2 = ()

                            # code downside is used to delete tracking box in a rough way
                            bbox = result['box_corner']
                            x_max,x_min = max(bbox[:,0]),min(bbox[:,0])
                            y_max,y_min = max(bbox[:,1]),min(bbox[:,1])
                            z_max,z_min = max(bbox[:,2]),min(bbox[:,2])

                            box_mask2 = (x_min < pcs_glo[:,0]) & (pcs_glo[:,0] < x_max) &\
                                        (y_min < pcs_glo[:,1]) & (pcs_glo[:,1] < y_max) &\
                                        (z_min < pcs_glo[:,2]) & (pcs_glo[:,2] < z_max)
                            tracking_mask = tracking_mask | box_mask2

                            bbox4d = np.concatenate((result['box_corner'], np.ones((8, 1))), axis=1)

                            bbox4d = np.dot(Trans_i,bbox4d.T).T
                            bbox2d = project_points_to_image(bbox4d)
                            image, _ = draw_box3d_image(image, bbox2d[:,:-1], result['id'])

                            np_center = np.concatenate([result['center'],[1]])
                            center_lidar = np.dot(Trans_i,np_center.T).T
                            marker = init_bbox_marker(center_lidar[:-1],result['wxyz'],result['size'],[0,1,0],id)
                            boxmsg.markers.append(marker)
                            id += 1
                        cv2.imshow("4dbox", image)
                        cv2.waitKey(1)

                #publish bbox to rviz, including detection bbox and tracking bbox
                try:
                    bboxpuber.publish(boxmsg)
                except:
                    print('publish error')

                # Stream results
                im0 = annotator.result()

                viz(im0, center_list, tp_dict)

                #publish the final PointCloud msg that used for SLAM, 3 masks optional:target_mask, all_box_mask, tracking_mask
                #target mask: delete all points in visual cone,
                #all_box_mask: delete points in 3d detection box,
                #tracking_mask: delete points in tracking box,
                if sum(tracking_mask): 
                    env_mask = ~tracking_mask
                    # env_mask = all_box_mask
                    env_index = np.arange(len(cam_points2d))[env_mask]
                    env_points = cam_pcs3d[env_index]
                    env_points_lf = pcs_lf[env_index]
                    env_msg = create_pointcloud2_np(env_points,env_points_lf,other_pcs)
                    env_msg.header.stamp = timestamp2
                    envpuber.publish(env_msg)
                else:
                    env_msg = create_pointcloud2_np(cam_pcs3d,pcs_lf,other_pcs)
                    env_msg.header.stamp = timestamp2
                    envpuber.publish(env_msg)
                    print('mask failed!')
                    
                
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
            
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    global WHOLE_TIME,T_N
    T_N = T_N + 1
    WHOLE_TIME = WHOLE_TIME + np.array([t[3],t[4],t[5],t[6]])
    MEANT = WHOLE_TIME/T_N
    print(f'MEAN!! project:{MEANT[0]:.1f}, delete:{MEANT[1]:.1f}, tracking:{MEANT[2]:.1f}ms, total:{MEANT[3]:.1f}, TIMES:{T_N}')
    LOGGER.info(f'''Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms Project, %.1fms Delete, %1fms Tracking, %1fms in total 
%1fms in Trans per image at shape {(1, 3, *imgsz)}''' % t)

    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)




def multi_callback(subscriber_lidar,subscriber_camera):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(subscriber_camera, desired_encoding='bgr8')
    run(model=modelx,image2=cv_image, lidar_msg=subscriber_lidar)


def main():
    # check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    rospy.init_node('image', anonymous=True)
    device = select_device('0')
    global modelx
    modelx = DetectMultiBackend(ROOT / 'yolov5s.pt', device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
    config = yaml.load(open('config/nusc_config.yaml', 'r'), Loader=yaml.Loader)
    global polyloader,polytracker,track_result
    track_result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    polyloader = NuScenesloader(config)
    polytracker = Tracker(config=polyloader.config)
    global puber,markerpuber,envpuber,bboxpuber
    puber = rospy.Publisher("filter_pcs",PointCloud2,queue_size=10)
    markerpuber = rospy.Publisher('centers',MarkerArray,queue_size=10)
    bboxpuber = rospy.Publisher('bboxs',MarkerArray,queue_size=10)
    envpuber = rospy.Publisher("env_pcs",PointCloud2,queue_size=10)


    subscriber_lidar = message_filters.Subscriber('/hesai/pandar',PointCloud2)
    subscriber_camera = message_filters.Subscriber("/camera/color/image_raw",Image)
    global tf_listener
    tf_listener = tf.TransformListener()

  
    sync = message_filters.ApproximateTimeSynchronizer([subscriber_lidar,subscriber_camera],10,0.1)

    sync.registerCallback(multi_callback)

    rospy.spin()


if __name__ == '__main__':
    # opt = parse_opt()
    main()
