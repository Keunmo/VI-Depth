import torch
import numpy as np

from modules.midas.midas_net_custom import MidasNet_small_videpth
from modules.estimator import LeastSquaresEstimator
from modules.interpolator import Interpolator2D

import modules.midas.transforms as transforms
import modules.midas.utils as utils
from pathlib import Path
from PIL import Image

class VIDepth(object):
    def __init__(self, depth_predictor, nsamples, sml_model_path, 
                min_pred, max_pred, min_depth, max_depth, device):

        # get transforms
        model_transforms = transforms.get_transforms(depth_predictor, "void", str(nsamples))
        self.depth_model_transform = model_transforms["depth_model"]
        self.ScaleMapLearner_transform = model_transforms["sml_model"]

        # define depth model
        if depth_predictor == "dpt_beit_large_512":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512")
        elif depth_predictor == "dpt_swin2_large_384":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_SwinV2_L_384")
        elif depth_predictor == "dpt_large":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        elif depth_predictor == "dpt_hybrid":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        elif depth_predictor == "dpt_swin2_tiny_256":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_SwinV2_T_256")
        elif depth_predictor == "dpt_levit_224":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_LeViT_224")
        elif depth_predictor == "midas_small":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        else:
            self.DepthModel = None

        # define SML model
        self.ScaleMapLearner = MidasNet_small_videpth(
            path=sml_model_path,
            min_pred=min_pred,
            max_pred=max_pred,
        )

        # depth prediction ranges
        self.min_pred, self.max_pred = min_pred, max_pred

        # depth evaluation ranges
        self.min_depth, self.max_depth = min_depth, max_depth

        # eval mode
        self.DepthModel.eval()
        self.DepthModel.to(device)

        # eval mode
        self.ScaleMapLearner.eval()
        self.ScaleMapLearner.to(device)


    def run(self, input_image, input_sparse_depth, validity_map, device):

        input_height, input_width = np.shape(input_image)[0], np.shape(input_image)[1]
        
        sample = {"image" : input_image}
        sample = self.depth_model_transform(sample)
        im = sample["image"].to(device)

        input_sparse_depth_valid = (input_sparse_depth < self.max_depth) * (input_sparse_depth > self.min_depth)
        if validity_map is not None:
            input_sparse_depth_valid *= validity_map.astype(bool)

        input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
        input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
        input_sparse_depth = 1.0 / input_sparse_depth  # look here, we are converting depth to inverse depth

        # run depth model
        with torch.no_grad():
            depth_pred = self.DepthModel.forward(im.unsqueeze(0))
            depth_pred = (
                torch.nn.functional.interpolate(
                    depth_pred.unsqueeze(1),
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # global scale and shift alignment
        GlobalAlignment = LeastSquaresEstimator(
            estimate=depth_pred,  # this is inverse depth by depth estimation model
            target=input_sparse_depth,  # this is inverse metric sparse depth 
            valid=input_sparse_depth_valid
        )
        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        GlobalAlignment.clamp_min_max(clamp_min=self.min_pred, clamp_max=self.max_pred)
        int_depth = GlobalAlignment.output.astype(np.float32)

        # interpolation of scale map
        assert (np.sum(input_sparse_depth_valid) >= 3), "not enough valid sparse points"
        ScaleMapInterpolator = Interpolator2D(
            pred_inv = int_depth,
            sparse_depth_inv = input_sparse_depth,
            valid = input_sparse_depth_valid,
        )
        ScaleMapInterpolator.generate_interpolated_scale_map(
            interpolate_method='linear', 
            fill_corners=False
        )
        int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
        int_scales = utils.normalize_unit_range(int_scales)

        sample = {"image" : input_image, "int_depth" : int_depth, "int_scales" : int_scales, "int_depth_no_tf" : int_depth}
        sample = self.ScaleMapLearner_transform(sample)
        x = torch.cat([sample["int_depth"], sample["int_scales"]], 0)
        x = x.to(device)
        d = sample["int_depth_no_tf"].to(device)

        # run SML model
        with torch.no_grad():
            sml_pred, sml_scales = self.ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
            sml_pred = (
                torch.nn.functional.interpolate(
                    sml_pred,
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        output = {
            "ga_depth"  : int_depth, 
            "sml_depth" : sml_pred, 
        }
        return output


if __name__ == "__main__":
    method = VIDepth(
        depth_predictor="dpt_beit_large_512", 
        nsamples=150, 
        sml_model_path="weights/sml_model.dpredictor.dpt_beit_large_512.nsamples.150.ckpt", 
        min_pred=0.1, 
        max_pred=15.0,
        min_depth=0.2, 
        max_depth=8.0, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    input_prefix = Path("input")
    output_prefix = Path("output")

    gt_dep_prefix = Path("ground_truth")
    gt_spdep_prefix = Path("sparse_depth")
    rgb_prefix = Path("image")

    ga_dep_prefix = Path("ga_depth")
    sml_dep_prefix = Path("sml_depth")

    scene = "classroom4"
    imname = "1552695831.4496.png"

    gt_rgb1 = utils.read_image((input_prefix / scene / rgb_prefix / imname).as_posix())
    gt_dep1 = np.array(Image.open((input_prefix / scene / gt_dep_prefix / imname).as_posix()), dtype=np.float32) / 256.0
    gt_spdep1 = np.array(Image.open((input_prefix / scene / gt_spdep_prefix / imname).as_posix()), dtype=np.float32) / 256.0

    validity_map = np.array(Image.open((input_prefix / scene / "validity_map" / imname).as_posix()), dtype=np.float32)
    assert(np.all(np.unique(validity_map) == [0, 256]))
    validity_map[validity_map <= 0] = 1

    output = method.run(
        input_image=gt_rgb1,
        input_sparse_depth=gt_spdep1,
        validity_map=validity_map,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )