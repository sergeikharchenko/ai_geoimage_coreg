import torch
import numpy as np
from kornia.feature import LoFTR

# Ensure expandable segments is set for memory optimization
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class BaseMatcher:
    def __init__(self, device):
        self.device = device

    def match(self, img1, img2):
        raise NotImplementedError

class LoFTRMatcher(BaseMatcher):
    def __init__(self, device, pretrained='outdoor'):
        super().__init__(device)
        print("Initializing LoFTR...")
        self.model = LoFTR(pretrained=pretrained).to(device).eval()

    def match(self, t_hex, t_ref, conf_thr=0.75):
        input_dict = {"image0": t_hex, "image1": t_ref}
        with torch.no_grad():
            res = self.model(input_dict)
        
        kpts0 = res['keypoints0'].cpu().numpy()
        kpts1 = res['keypoints1'].cpu().numpy()
        conf = res['confidence'].cpu().numpy()
        
        valid = conf > conf_thr
        return kpts0[valid], kpts1[valid], conf[valid]

class SuperGlueWrapper(BaseMatcher):
    def __init__(self, device, weights_path=None):
        super().__init__(device)
        print("Initializing SuperPoint+SuperGlue...")
        
        # NOTE: In a real package, you might vendor the SuperGlue models 
        # into a subfolder 'models/' and import them here.
        # For this example, we assume the standard SuperGlue repo is in python path
        # or we use a placeholder if not present.
        try:
            from models.matching import Matching
        except ImportError:
            raise ImportError("SuperGlue models not found. Please clone SuperGluePretrainedNetwork or add it to PYTHONPATH.")

        config = {
            'superpoint': {
                'nms_radius': 3,
                'keypoint_threshold': 0.005,
                'max_keypoints': 2048,
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.model = Matching(config).eval().to(device)

    def match(self, t_hex, t_ref, conf_thr=0.9):
        # SuperPoint expects normalized tensors
        input_dict = {"image0": t_hex, "image1": t_ref}
        
        with torch.no_grad():
            pred = self.model(input_dict)
            
        matches = pred['matches0'][0].cpu().numpy()
        conf = pred['matching_scores0'][0].cpu().numpy()
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        
        valid = (matches > -1) & (conf > conf_thr)
        
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        
        return mkpts0, mkpts1, mconf