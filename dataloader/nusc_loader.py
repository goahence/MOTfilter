"""
dataloader of NuScenes dataset
Obtain the observation information(detection) of each frame iteratively
--------ATTENTION: Detector files must be in chronological order-------
"""

import pdb
import numpy as np
from utils.io import load_file
from data.script.NUSC_CONSTANT import *
from pre_processing import dictdet2array, arraydet2box, blend_nms

threshold = {0: 0.14, 3: 0.16, 1: 0, 2: 0.16, 5: 0.1, 6: 0, 4: 0.7}


class NuScenesloader:
    def __init__(self, config):
        """
        :param detection_path: path of order detection file
        :param first_token_path: path of first frame token for each seq
        :param config: dict, hyperparameter setting
        """
        # detector -> {sample_token:[{det1_info}, {det2_info}], ...}， check the detailed "det_info" at nuscenes.org
        # self.detector = load_file(detection_path)["results"]
        self.all_sample_token = ['0']
        # self.seq_first_token = load_file(first_token_path)
        self.seq_first_token = ['0']
        self.config, self.data_info = config, {}
        self.SF_thre, self.NMS_thre = config['preprocessing']['SF_thre'], config['preprocessing']['NMS_thre']
        self.NMS_type, self.NMS_metric = config['preprocessing']['NMS_type'], config['preprocessing']['NMS_metric']
        self.seq_id = self.frame_id = 0
        self.result = None

    def __getitem__(self, item) -> dict:
        """
        data_info(dict): {
            'is_first_frame': bool
            'timestamp': int
            'sample_token': str
            'seq_id': int
            'frame_id': int
            'has_velo': bool
            'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
            'np_dets_bottom_corners': np.array, [det_num, 4, 2]
            'box_dets': np.array[NuscBox], [det_num]
            'no_dets': bool, corner case,
            'det_num': int,
        }
        """
        curr_token = '1'
        # ori_dets = self.result

        # assign seq and frame id
        if curr_token in self.seq_first_token:
            self.seq_id += 1
            self.frame_id = 1
        else: self.frame_id += 1

        # all categories are blended together and sorted by detection score
        # list_dets, np_dets = dictdet2array(ori_dets, 'translation', 'size', 'velocity', 'rotation',
        #                                    'detection_score', 'detection_name')

        pdb.set_trace()
        # Score Filter based on category-specific thresholds
        # np_dets = np.array([det for det in list_dets if det[-2] > self.SF_thre[det[-1]]])
        np_dets = self.result
        list_dets = {'0','1'}

        # NMS, "blend" ref to blend all categories together during NMS
        if len(np_dets) != 0:
            box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
            assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners)
            tmp_infos = {'np_dets': np_dets, 'np_dets_bottom_corners': np_dets_bottom_corners}
            keep = globals()[self.NMS_type](box_infos=tmp_infos, metrics=self.NMS_metric, thre=self.NMS_thre)
            keep_num = len(keep)
        # corner case, no det left
        else: keep = keep_num = 0

        print(f"\n Total {len(list_dets) - keep_num} bboxes are filtered; "
              f"{len(list_dets) - len(np_dets)} during SF, "
              f"{len(np_dets) - keep_num} during NMS, "
              f"Still {keep_num} bboxes left. "
              f"seq id {self.seq_id}, frame id {self.frame_id}, "
              f"Total frame id ?.")

        # Available information for the current frame
        data_info = {
            'is_first_frame': curr_token in self.seq_first_token,
            'timestamp': item,
            'sample_token': curr_token,
            'seq_id': self.seq_id,
            'frame_id': self.frame_id,
            'has_velo': self.config['basic']['has_velo'],
            'np_dets': np_dets[keep] if keep_num != 0 else np.zeros(0),
            'np_dets_bottom_corners': np_dets_bottom_corners[keep] if keep_num != 0 else np.zeros(0),
            'box_dets': box_dets[keep] if keep_num != 0 else np.zeros(0),
            'no_dets': keep_num == 0,
            'det_num': keep_num,
        }
        return data_info
    
    def get_result(self):
        # curr_token = '1'
        # ori_dets = self.np_det

        # assign seq and frame id
        # if curr_token in self.seq_first_token:
        #     self.seq_id += 1
        #     self.frame_id = 1
        # else: self.frame_id += 1

        # # all categories are blended together and sorted by detection score
        # list_dets, np_dets = dictdet2array(ori_dets, 'translation', 'size', 'velocity', 'rotation',
        #                                    'detection_score', 'detection_name')

        # # Score Filter based on category-specific thresholds
        # np_dets = np.array([det for det in list_dets if det[-2] > self.SF_thre[det[-1]]])
        
        np_dets = self.result
        ori_num = len(np_dets)

        # score filter
        # np_dets = np_dets[np_dets[:,-2] > 0.7]
        np_dets = np.array([item for item in np_dets if item[-2] > threshold[item[-1]]])

        # NMS, "blend" ref to blend all categories together during NMS
        if len(np_dets) != 0:
            box_dets, np_dets_bottom_corners, np_dets_norm_corners = arraydet2box(np_dets)
            assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners) == len(np_dets_norm_corners)
            tmp_infos = {'np_dets': np_dets,
                         'np_dets_bottom_corners': np_dets_bottom_corners,
                         'np_dets_norm_corners': np_dets_norm_corners,}
            keep = globals()[self.NMS_type](box_infos=tmp_infos, metrics=self.NMS_metric, thre=self.NMS_thre)
            keep_num = len(keep)
        # corner case, no det left
        else: keep = keep_num = 0

        print(f"\n Total {ori_num - keep_num} bboxes are filtered; "
              f"{ori_num - len(np_dets)} during SF, "
              f"{len(np_dets) - keep_num} during NMS, "
              f"Still {keep_num} bboxes left. "
              f"seq id {self.seq_id}, frame id {self.frame_id}")

        # if len(np_dets) != 0:
        #     box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
        # else:
        #     box_dets, np_dets_bottom_corners = 0,0
            
        # Available information for the current frame

        data_info = {
            'is_first_frame': self.frame_id == 1,
            'timestamp': 0,
            'sample_token': self.frame_id,
            'seq_id': self.seq_id,
            'frame_id': self.frame_id,
            'has_velo': self.config['basic']['has_velo'],
            'np_dets': np_dets[keep] if keep_num != 0 else np.zeros(0),
            'np_dets_norm_corners': np_dets_norm_corners[keep] if keep_num != 0 else np.zeros(0),
            'np_dets_bottom_corners': np_dets_bottom_corners[keep] if keep_num != 0 else np.zeros(0),
            'box_dets': box_dets[keep] if keep_num != 0 else np.zeros(0),
            'no_dets': keep_num == 0,
            'det_num': keep_num,
        }
        # data_info = {
        #     'is_first_frame': self.frame_id == 1,
        #     'timestamp': 0,
        #     'sample_token': self.frame_id,
        #     'seq_id': self.seq_id,
        #     'frame_id': self.frame_id,
        #     'has_velo': self.config['basic']['has_velo'],
        #     'np_dets': np_dets,
        #     'np_dets_bottom_corners': np_dets_bottom_corners,
        #     'box_dets': box_dets,
        #     'no_dets': len(np_dets) == 0,
        #     'det_num': len(np_dets),
        # }
        # self.data_info = data_info
        return data_info

    def update(self, result):
        self.frame_id += 1
        self.result = result

    def __len__(self) -> int:
        return len(self.all_sample_token)
