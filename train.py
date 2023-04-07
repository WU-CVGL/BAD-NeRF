import torch

from nerf import *
import optimize_pose_linear, optimize_pose_cubic
import torchvision.transforms.functional as torchvision_F

import matplotlib.pyplot as plt

from metrics import compute_img_metric
import novel_view_test


def train():
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
    print_file = os.path.join(basedir, expname, 'print.txt')
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

    if args.load_weights:
        if args.linear:
            print('Linear Spline Model Loading!')
            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
        else:
            print('Cubic Spline Model Loading!')
            model = optimize_pose_cubic.Model(poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3)
        graph = model.build_network(args)
        optimizer, optimizer_se3 = model.setup_optimizer(args)
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))  # here
        graph_ckpt = torch.load(path)
        graph.load_state_dict(graph_ckpt['graph'])
        optimizer.load_state_dict(graph_ckpt['optimizer'])
        optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
        global_step = graph_ckpt['global_step']

    else:
        if args.linear:
            low, high = 0.0001, 0.005
            rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            poses_start_se3 = poses_start_se3 + rand

            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
        else:
            low, high = 0.0001, 0.01
            rand1 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            rand2 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            rand3 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            poses_se3_1 = poses_start_se3 + rand1
            poses_se3_2 = poses_start_se3 + rand2
            poses_se3_3 = poses_start_se3 + rand3

            model = optimize_pose_cubic.Model(poses_start_se3, poses_se3_1, poses_se3_2, poses_se3_3)

        graph = model.build_network(args)  # nerf, nerf_fine, forward
        optimizer, optimizer_se3 = model.setup_optimizer(args)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0
    if not args.load_weights:
        global_step = start
    global_step_ = global_step
    threshold = N_iters + 1

    poses_num = poses.shape[0]

    for i in trange(start, threshold):
    ### core optimization loop ###
        i = i+global_step_
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        img_idx = torch.randperm(images.shape[0])

        if (i % args.i_img == 0 or i % args.i_novel_view == 0) and i > 0:
            ret, ray_idx, spline_poses, all_poses = graph.forward(i, img_idx, poses_num, H, W, K, args)
        else:
            ret, ray_idx, spline_poses = graph.forward(i, img_idx, poses_num, H, W, K, args)

        # get image ground truth
        target_s = images[img_idx].reshape(-1, H * W, 3)
        target_s = target_s[:, ray_idx]
        target_s = target_s.reshape(-1, 3)

        # average
        shape0 = img_idx.shape[0]
        interval = target_s.shape[0] // shape0
        rgb_list = []
        extras_list = []
        rgb_ = 0
        extras_ = 0

        for j in range(0, shape0 * args.deblur_images):
            rgb_ += ret['rgb_map'][j * interval:(j + 1) * interval]
            if 'rgb0' in ret:
                extras_ += ret['rgb0'][j * interval:(j + 1) * interval]
            if (j + 1) % args.deblur_images == 0:
                rgb_ = rgb_ / args.deblur_images
                rgb_list.append(rgb_)
                rgb_ = 0
                if 'rgb0' in ret:
                    extras_ = extras_ / args.deblur_images
                    extras_list.append(extras_)
                    extras_ = 0

        rgb_blur = torch.stack(rgb_list, 0)
        rgb_blur = rgb_blur.reshape(-1, 3)

        if 'rgb0' in ret:
            extras_blur = torch.stack(extras_list, 0)
            extras_blur = extras_blur.reshape(-1, 3)

        # backward
        optimizer_se3.zero_grad()
        optimizer.zero_grad()
        img_loss = img2mse(rgb_blur, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in ret:
            img_loss0 = img2mse(extras_blur, target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()

        optimizer.step()
        optimizer_se3.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = 0.01
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose
        ###############################

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item()}")
            with open(print_file, 'a') as outfile:
                outfile.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item()}\n")

        if i < 10:
            print('coarse_loss:', img_loss0.item())
            with open(print_file, 'a') as outfile:
                outfile.write(f"coarse loss: {img_loss0.item()}\n")

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_se3': optimizer_se3.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_img == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                if args.deblur_images % 2 == 0:
                    i_render = torch.arange(i_train.shape[0]) * (args.deblur_images+1) + args.deblur_images // 2
                else:
                    i_render = torch.arange(i_train.shape[0]) * args.deblur_images + args.deblur_images // 2
                imgs_render = render_image_test(i, graph, all_poses[i_render], H, W, K, args)
            mse_render = compute_img_metric(imgs_sharp, imgs_render, 'mse')
            psnr_render = compute_img_metric(imgs_sharp, imgs_render, 'psnr')
            ssim_render = compute_img_metric(imgs_sharp, imgs_render, 'ssim')
            lpips_render = compute_img_metric(imgs_sharp, imgs_render, 'lpips')
            with open(test_metric_file, 'a') as outfile:
                outfile.write(f"iter{i}: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"
                              f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_video_test(i, graph, render_poses, H, W, K, args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if args.novel_view and i % args.i_novel_view == 0 and i > 0:
            # Turn on novel view testing mode
            i_ = torch.arange(0, images.shape[0], args.llffhold-1)
            poses_test_se3_ = graph.se3.weight[i_,:6]
            model_test = novel_view_test.Model(poses_test_se3_, graph)
            graph_test = model_test.build_network(args)
            optimizer_test = model_test.setup_optimizer(args)
            for j in range(args.N_novel_view):
                ret_sharp, ray_idx_sharp, poses_sharp = graph_test.forward(i, img_idx, poses_num, H, W, K, args, novel_view=True)
                target_s_novel = images_novel.reshape(-1, H*W, 3)[:, ray_idx_sharp]
                target_s_novel = target_s_novel.reshape(-1, 3)
                loss_sharp = img2mse(ret_sharp['rgb_map'], target_s_novel)
                psnr_sharp = mse2psnr(loss_sharp)
                if 'rgb0' in ret_sharp:
                    img_loss0 = img2mse(ret_sharp['rgb0'], target_s_novel)
                    loss_sharp = loss_sharp + img_loss0
                if j%100==0:
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
                imgs_render_novel = render_image_test(i, graph, poses_sharp, H, W, K, args, novel_view=True)

                mse_render = compute_img_metric(images_novel, imgs_render_novel, 'mse')
                psnr_render = compute_img_metric(images_novel, imgs_render_novel, 'psnr')
                ssim_render = compute_img_metric(images_novel, imgs_render_novel, 'ssim')
                lpips_render = compute_img_metric(images_novel, imgs_render_novel, 'lpips')
                with open(test_metric_file_novel, 'a') as outfile:
                    outfile.write(f"iter{i}: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"
                                  f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")

        if i % args.N_iters == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                path_pose = os.path.join(basedir, expname)
                i_render_pose = torch.arange(i_train.shape[0]) * args.deblur_images + args.deblur_images // 2
                render_poses_final = all_poses[i_render_pose]
                save_render_pose(render_poses_final, path_pose)

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
