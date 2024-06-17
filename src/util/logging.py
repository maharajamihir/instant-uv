import torch
import cv2
import numpy as np
import matplotlib as mpl
from torch.utils.tensorboard import SummaryWriter

class TensorboardLoggingDiligent:
    def __init__(
        self,
        device,
        available_views_light_pairs_train,
        available_views_light_pairs_test,
        **kwargs
    ):
        self.tensorboard = SummaryWriter()
        self.device = device

        # Construct indices for rendering
        n_views_train = len({i[0] for i in available_views_light_pairs_train})
        n_views_test = len({i[0] for i in available_views_light_pairs_test})

        # For rendering, choose one evenly spaced light indices per view
        n_view_lights_train = len(available_views_light_pairs_train)
        step_train = len(available_views_light_pairs_train) // n_views_train
        self.indices_rendering_train = torch.arange(0, n_view_lights_train, step_train) +\
            torch.arange(0, n_views_train)
        
        n_view_lights_test = len(available_views_light_pairs_test)
        step_test = len(available_views_light_pairs_test) // n_views_test
        self.indices_rendering_test = torch.arange(0, n_view_lights_test, step_test) +\
            torch.arange(0, n_views_test)
        
        color_dict = {
            'red': [
                (0.0, 0.0, 2/256),
                (1.0, 232/256, 1.0)
            ],
            'green': [
                (0.0, 1.0, 15/256),
                (1.0, 39/256, 0.0)
            ],
            'blue': [
                (0.0, 0.0, 50/256),
                (1.0, 14/256, 0.0)
            ]
            }
        self.error_colormap = mpl.colors.LinearSegmentedColormap('my_cmap', color_dict)
        
        print("Done with initialization")

    def zero_losses_epoch(self):
        self.loss_epoch_full = torch.zeros([]).to(self.device)
        self.loss_epoch_data = torch.zeros([]).to(self.device)
        self.loss_epoch_reg = torch.zeros([]).to(self.device)

        self.n_batches = 0

    def accumulate_losses(self, losses):
        self.loss_epoch_full += losses['loss']
        self.loss_epoch_data += losses['loss_data']

        if losses['loss_reg'] is not None:
            self.loss_epoch_reg += losses['loss_reg']
        else:
            self.loss_epoch_reg = None 

        self.n_batches += 1

    def log_losses(self, epoch):
        self.tensorboard.add_scalar('Train Loss', self.loss_epoch_full / self.n_batches, epoch)
        self.tensorboard.add_scalar('Data Loss', self.loss_epoch_data / self.n_batches, epoch)
        if self.loss_epoch_reg is not None:
            self.tensorboard.add_scalar('Reg Loss', self.loss_epoch_reg / self.n_batches, epoch)

    def log_images(self, renderings_train, renderings_test, epoch, metrics_train=None, metrics_test=None):
        # Render the images
        print("Writing images to tensorboard")
        self.log_images_dataset(
            renderings=renderings_train,
            indices=self.indices_rendering_train,
            name_dataset_caption='Train Dataset',
            epoch=epoch,
            metrics=metrics_train
        )
        self.log_images_dataset(
            renderings=renderings_test,
            indices=self.indices_rendering_test,
            name_dataset_caption='Test Dataset',
            epoch=epoch,
            metrics=metrics_test,
        )

    def log_metrics(self, metrics_train, metrics_test, epoch):
        """
        """

        # Log metrics
        self.tensorboard.add_scalars(
            "PSNR linear (mean over the per image values)",
            {'Train': np.mean(metrics_train['psnrs_linear']), 'Test': np.mean(metrics_test['psnrs_linear'])},
            epoch
        )
        self.tensorboard.add_scalars(
            "PSNR sRGB (mean over the per image values)",
            {'Train': np.mean(metrics_train['psnrs_srgb']), 'Test': np.mean(metrics_test['psnrs_srgb'])},
            epoch
        )
        self.tensorboard.add_scalars(
            "SSIM",
            {'Train': np.mean(metrics_train['ssims']), 'Test': np.mean(metrics_test['ssims'])},
            epoch
        )
        self.tensorboard.add_scalars(
            "LPIPS",
            {'Train': torch.mean(metrics_train['lpips']), 'Test': torch.mean(metrics_test['lpips'])},
            epoch
        )

    def log_images_dataset(
        self,
        renderings,
        indices,
        name_dataset_caption,
        epoch,
        metrics=None,
    ):
        images_linear = []
        images_srgb = []
        error_maps_srgb = []

        for i in indices:
            rendering_linear = renderings['images_linear'][i].numpy()
            image_gt_linear = renderings['images_gt_linear'][i].numpy()
            rendering_srgb = renderings['images_srgb'][i].numpy()
            image_gt_srgb = renderings['images_gt_srgb'][i].numpy()

            # Compute error maps
            errors_srgb = np.linalg.norm(rendering_srgb - image_gt_srgb, axis=-1)
            errors_srgb = self.error_colormap(errors_srgb / errors_srgb.max())
            error_maps_srgb.append(torch.from_numpy(errors_srgb[..., :3]))

            psnr_linear = 0 if metrics is None else metrics['psnrs_linear'][i]
            psnr_srgb = 0 if metrics is None else metrics['psnrs_srgb'][i]
            ssim = 0 if metrics is None else metrics['ssims'][i]
            lpips_cur = 0 if metrics is None else metrics['lpips'][i]

            # Add linear images
            rendering_linear = cv2.putText(
                rendering_linear,
                f"PSNR: {psnr_linear:.1f}  -  SSIM: {ssim:.3f}  -  LPIPS: {lpips_cur:,.4f}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255),
                thickness=1
            )
            rendering_linear = torch.from_numpy(rendering_linear)
            image_gt_linear = cv2.putText(
                image_gt_linear,
                "GT",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255),
                thickness=1
            )
            image_gt_linear = torch.from_numpy(image_gt_linear)

            cur_image = torch.cat([
                torch.ones(20, rendering_linear.shape[1], 3),
                rendering_linear,
                torch.ones(5, rendering_linear.shape[1], 3),
                image_gt_linear,
                torch.ones(20, rendering_linear.shape[1], 3)
                ],
                dim=0
            )
            cur_image = torch.cat([torch.ones(cur_image.shape[0], 10, 3), cur_image, torch.ones(cur_image.shape[0], 10, 3)], dim=1)
            images_linear.append(cur_image)

            # Add srgb images
            rendering_srgb = cv2.putText(
                rendering_srgb,
                f"PSNR: {psnr_srgb:.1f}  -  SSIM: {ssim:.3f}  -  LPIPS: {lpips_cur:,.4f}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255),
                thickness=1
            )
            rendering_srgb = torch.from_numpy(rendering_srgb)
            image_gt_srgb = cv2.putText(
                image_gt_srgb,
                "GT",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255),
                thickness=1
            )
            image_gt_srgb = torch.from_numpy(image_gt_srgb)

            cur_image = torch.cat([
                torch.ones(20, rendering_srgb.shape[1], 3),
                rendering_srgb,
                torch.ones(5, rendering_srgb.shape[1], 3),
                image_gt_srgb,
                torch.ones(20, rendering_srgb.shape[1], 3)
                ],
                dim=0
            )
            cur_image = torch.cat([torch.ones(cur_image.shape[0], 10, 3), cur_image, torch.ones(cur_image.shape[0], 10, 3)], dim=1)
            images_srgb.append(cur_image)


        self.tensorboard.add_images(
            f'Renderings linear {name_dataset_caption} (gt bottom)',
            torch.stack(images_linear, dim=0).permute(0, 3, 1, 2),
            epoch
        )
        self.tensorboard.add_images(
            f'Renderings sRGB {name_dataset_caption} (gt bottom)',
            torch.stack(images_srgb, dim=0).permute(0, 3, 1, 2),
            epoch
        )
        self.tensorboard.add_images(
            f"Pixelwise L2 Color Error - sRGB {name_dataset_caption} (Normalized by individual maximum per image)",
            torch.stack(error_maps_srgb, dim=0).permute(0, 3, 1, 2),
            epoch
        )
