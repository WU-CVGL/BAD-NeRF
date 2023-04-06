import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm, trange
import os
import imageio


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2se = lambda x, y : (x - y) ** 2
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))  # logab = logcb / logca
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to8b_tensor = lambda x : (255*torch.clip(x,0,1)).type(torch.int)


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def load_imgs(path):
    imgfiles = [os.path.join(path, f) for f in sorted(os.listdir(path)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = imgs.astype(np.float32)
    imgs = torch.tensor(imgs).cuda()

    return imgs


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def render_video_test(i_, graph, render_poses, H, W, K, args):
    rgbs = []
    disps = []
    # t = time.time()
    for i, pose in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        # t = time.time()
        pose = pose[None, :3, :4]
        ret = graph.render_video(i_, pose[:3, :4], H, W, K, args)
        rgbs.append(ret['rgb_map'].cpu().numpy())
        disps.append(ret['disp_map'].cpu().numpy())
        if i==0:
            print(ret['rgb_map'].shape, ret['disp_map'].shape)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def render_image_test(i, graph, render_poses, H, W, K, args, novel_view=False, need_depth=False):
    if novel_view:
        img_dir = os.path.join(args.basedir, args.expname, 'img_novel_{:06d}'.format(i))
    else:
        img_dir = os.path.join(args.basedir, args.expname, 'img_test_{:06d}'.format(i))
    os.makedirs(img_dir, exist_ok=True)
    imgs =[]

    for j, pose in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        # t = time.time()
        pose = pose[None, :3, :4]
        ret = graph.render_video(i, pose[:3, :4], H, W, K, args)
        imgs.append(ret['rgb_map'])
        rgbs = ret['rgb_map'].cpu().numpy()
        rgb8 = to8b(rgbs)
        imageio.imwrite(os.path.join(img_dir, 'rgb_{:03d}.png'.format(j)), rgb8)
        if need_depth:
            depths = ret['disp_map'].cpu().numpy()
            depths_ = depths/np.max(depths)
            depth8 = to8b(depths_)
            imageio.imwrite(os.path.join(img_dir, 'depth_{:03d}.png'.format(j)), depth8)
    imgs = torch.stack(imgs, 0)
    return imgs


def init_weights(linear):
    # use Xavier init instead of Kaiming init
    torch.nn.init.kaiming_normal_(linear.weight)
    torch.nn.init.zeros_(linear.bias)


def init_nerf(nerf):
    for linear_pt in nerf.pts_linears:
        init_weights(linear_pt)

    for linear_view in nerf.views_linears:
        init_weights(linear_view)

    init_weights(nerf.feature_linear)

    init_weights(nerf.alpha_linear)

    init_weights(nerf.rgb_linear)

# Ray helpers only get specific rays
def get_specific_rays(i, j, K, c2w):
    # i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
    #                       torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    # i = i.t()
    # j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[..., :3, :3], -1)
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[..., :3, -1]
    return rays_o, rays_d


def save_render_pose(poses, path):
    poses_np = poses.cpu().detach().numpy()
    N = poses_np.shape[0]
    bottom = np.reshape([0., 0., 0., 1.], [1, 4])
    bottom_all = np.expand_dims(bottom, 0).repeat(N, axis=0)
    poses_Rt = np.concatenate([poses_np, bottom_all], 1)

    poses_txt = os.path.join(path, 'poses_render.txt')

    for j in range(poses_np.shape[0]):
        poses_flat = poses_Rt[j].reshape(16, 1).squeeze()
        for k in range(16):
            with open(poses_txt, 'a') as outfile:
                if k == 0:
                    outfile.write(f"pose{j} ")
                if k != 15:
                    outfile.write(f"{poses_flat[k]} ")
                if k == 15:
                    outfile.write(f"{poses_flat[k]}\n")
