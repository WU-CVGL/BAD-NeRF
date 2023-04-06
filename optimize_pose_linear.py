import torch.nn

import Spline
import nerf


class Model(nerf.Model):
    def __init__(self, se3_start, se3_end):
        super().__init__()
        self.start = se3_start
        self.end = se3_end

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.se3 = torch.nn.Embedding(self.start.shape[0], 6*2)

        start_end = torch.cat([self.start, self.end], -1)
        self.graph.se3.weight.data = torch.nn.Parameter(start_end)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim, self.optim_se3


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        self.pose_eye = torch.eye(3, 4)
        self.se3_start = None
        self.se3_end = None

    def get_pose(self, i, img_idx, args):
        se3_start = self.se3.weight[:, :6][img_idx]
        se3_end = self.se3.weight[:, 6:][img_idx]
        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_start.shape[0], 1)
        seg_pos_x = torch.arange(se3_start.shape[0]).reshape([se3_start.shape[0], 1]).repeat(1, args.deblur_images)

        se3_start = se3_start[seg_pos_x, :]
        se3_end = se3_end[seg_pos_x, :]

        spline_poses = Spline.SplineN_linear(se3_start, se3_end, pose_nums, args.deblur_images)
        return spline_poses
    
    def get_pose_even(self, i, img_idx, num):
        deblur_images_num = num+1
        se3_start = self.se3.weight[:, :6][img_idx]
        se3_end = self.se3.weight[:, 6:][img_idx]
        pose_nums = torch.arange(deblur_images_num).reshape(1, -1).repeat(se3_start.shape[0],1)
        seg_pos_x = torch.arange(se3_start.shape[0]).reshape([se3_start.shape[0], 1]).repeat(1, deblur_images_num)

        se3_start = se3_start[seg_pos_x, :]
        se3_end = se3_end[seg_pos_x, :]

        spline_poses = Spline.SplineN_linear(se3_start, se3_end, pose_nums, deblur_images_num)
        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses
