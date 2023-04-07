import torch

from nerf import *
import optimize_pose_linear, optimize_pose_cubic
import torchvision.transforms.functional as torchvision_F

import matplotlib.pyplot as plt

from metrics import compute_img_metric
import novel_view_test


def test():
    parser = config_parser()
    args = parser.parse_args()
    print('spline numbers: ', args.deblur_images)

    imgs_sharp_dir = os.path.join(args.datadir, 'images_test')
    imgs_sharp = load_imgs(imgs_sharp_dir)

    # Load data images and groundtruth
    K = None
    if args.dataset_type == 'llff':
        images_all, poses_start, bds_start, render_poses = load_llff_data(args.datadir, pose_state=None,
                                                                      factor=args.factor, recenter=True,
                                                                      bd_factor=.75, spherify=args.spherify)
        hwf = poses_start[0, :3, -1]

        # split train/val/test
        if args.novel_view:
            i_test = torch.arange(0, images_all.shape[0], args.llffhold)
        else:
            i_test = torch.tensor([100]).long()
        i_val = i_test
        i_train = torch.Tensor([i for i in torch.arange(int(images_all.shape[0])) if
                                (i not in i_test and i not in i_val)]).long()

        # train data
        images = images_all[i_train]
        # novel view data
        if args.novel_view:
            images_novel = images_all[i_test]
        # gt data
        imgs_sharp = imgs_sharp

        # get poses
        poses_end = poses_start
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = poses_start_se3
        poses_org = poses_start.repeat(args.deblur_images, 1, 1)
        poses = poses_org[:, :, :4]

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = torch.min(bds_start) * .9
            far = torch.max(bds_start) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    test_metric_file_novel = os.path.join(basedir, expname, 'test_metrics_novel.txt')
    # print_file = os.path.join(basedir, expname, 'print.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    if args.linear:
        print('Linear Spline Model Loading!')
        model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
    else:
        print('Cubic Spline Model Loading!')
        model = optimize_pose_cubic.Model(poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3)
    graph = model.build_network(args)
    optimizer, optimizer_se3 = model.setup_optimizer(args)
    path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))
    graph_ckpt = torch.load(path)
    graph.load_state_dict(graph_ckpt['graph'])
    optimizer.load_state_dict(graph_ckpt['optimizer'])
    optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
    global_step = graph_ckpt['global_step']

    if args.deblur_images % 2 == 0:
        all_poses = graph.get_pose_even(0, torch.arange(graph.se3.weight.shape[0]), args.deblur_images)
    else:
        all_poses = graph.get_pose(0, torch.arange(graph.se3.weight.shape[0]), args)
    # Turn on testing mode
    with torch.no_grad():
        if args.deblur_images % 2 == 0:
            i_render = torch.arange(i_train.shape[0]) * (args.deblur_images + 1) + args.deblur_images // 2
        else:
            i_render = torch.arange(i_train.shape[0]) * args.deblur_images + args.deblur_images // 2
        imgs_render = render_image_test(0, graph, all_poses[i_render], H, W, K, args)
    mse_render = compute_img_metric(imgs_sharp, imgs_render, 'mse')
    psnr_render = compute_img_metric(imgs_sharp, imgs_render, 'psnr')
    ssim_render = compute_img_metric(imgs_sharp, imgs_render, 'ssim')
    lpips_render = compute_img_metric(imgs_sharp, imgs_render, 'lpips')
    with open(test_metric_file, 'a') as outfile:
        outfile.write(f"test: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"

              f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")

    # Turn on novel view testing mode
    if args.novel_view:
        i_ = torch.arange(0, images.shape[0], args.llffhold - 1)
        poses_test_se3_ = graph.se3.weight[i_, :6]
        model_test = novel_view_test.Model(poses_test_se3_, graph)
        graph_test = model_test.build_network(args)
        optimizer_test = model_test.setup_optimizer(args)
        for j in range(args.N_novel_view):
            ret_sharp, ray_idx_sharp, poses_sharp = graph_test.forward(0, 0, 0, H, W, K, args,
                                                                       novel_view=True)
            target_s_novel = images_novel.reshape(-1, H * W, 3)[:, ray_idx_sharp]
            target_s_novel = target_s_novel.reshape(-1, 3)
            loss_sharp = img2mse(ret_sharp['rgb_map'], target_s_novel)
            psnr_sharp = mse2psnr(loss_sharp)
            if 'rgb0' in ret_sharp:
                img_loss0 = img2mse(ret_sharp['rgb0'], target_s_novel)
                loss_sharp = loss_sharp + img_loss0
            if j % 100 == 0:
                print(psnr_sharp.item(), loss_sharp.item())
            optimizer_test.zero_grad()
            loss_sharp.backward()
            optimizer_test.step()
            decay_rate_sharp = 0.01
            decay_steps_sharp = args.lrate_decay * 100
            new_lrate_novel = args.pose_lrate * (decay_rate_sharp ** (j / decay_steps_sharp))
            for param_group in optimizer_test.param_groups:
                if (j / decay_steps_sharp) <= 1.:
                    param_group['lr'] = new_lrate_novel * args.factor_pose_novel
        with torch.no_grad():
            imgs_render_novel = render_image_test(0, graph, poses_sharp, H, W, K, args, novel_view=True)

            mse_render = compute_img_metric(images_novel, imgs_render_novel, 'mse')
            psnr_render = compute_img_metric(images_novel, imgs_render_novel, 'psnr')
            ssim_render = compute_img_metric(images_novel, imgs_render_novel, 'ssim')
            lpips_render = compute_img_metric(images_novel, imgs_render_novel, 'lpips')
            with open(test_metric_file_novel, 'a') as outfile:
                outfile.write(f"novel view test: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"
                              f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")

    return 0


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    test()
