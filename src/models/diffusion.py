from utils import *
import baselines.simplex_noise
from scipy.ndimage.filters import gaussian_filter
import lpips
from scipy.ndimage import median_filter, percentile_filter, grey_dilation, grey_closing, maximum_filter, grey_opening
import torch
import lpips
import contextlib
import os
import sys
class Diffusion:
    def __init__(self, noise_steps, img_size, beta_start, beta_end, device):
        self.noise_steps = noise_steps
        
        self.beta = self.linear_noise_schedule(beta_start, beta_end).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def linear_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def noise_images(self, x, t, pyramid=False, simplex=False, discount = 0.8):
        """This method generates the latent representations at time t of the input images. Equation (1) in ANDi paper

        Parameters
        ----------
        x : tensor
            Input x_0
        t : tensor
            tensor containing the values for the time steps t
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        simplex : bool, optional
            flag that decides if simplex noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor, tensor
            The first tensor contains the latent representations, the second the noise used (needed for training)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        if simplex == True:
            slice_t = (torch.arange(1) + 10).long()
            tmp = torch.randn(
                (1, x.shape[0] * x.shape[1], self.img_size, self.img_size)
            )
            noise = baselines.simplex_noise.generate_simplex_noise(
                tmp, slice_t, in_channels=tmp.shape[1]
            )
            noise = noise.view(x.shape[0], x.shape[1], self.img_size, self.img_size).to(
                self.device
            )
        elif pyramid == True:
            noise = pyramid_noise_like(x.shape[0], x.shape[1], self.img_size, discount, x.device)
        else:
            noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def add_noise_images(self, x, t, noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
    
    def de_noise(self, x_t, t, predicted_noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        reconstructed_x0 = (x_t - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
        return reconstructed_x0
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def ddpm_mu_t(self, x, predicted_noise, t):
        """这个方法计算给定噪声的时刻t的高斯跃迁的平均值。纸上的符号是mu_q，或者mu_theta。

        Parameters
        ----------
        x : tensor
            包含x_t的张量
        predicted_noise : tensor
            包含噪声的张量(无论是预测的还是真实的)
        t : tensor
            包含时间步长的张量

        Returns
        -------
        tensor
            包含x_t-1的张量
        """
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        return (
            1
            / torch.sqrt(alpha)
            * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
        )

    def ddpm_mean_t(self, x, t, predicted_noise=None, x_0=None):
        """这个方法在给定预测噪声或原始样本x_0的情况下，计算高斯跃迁在时刻t的均值。
此外，当提供噪声来计算平均值时，像DDPM的原始实现一样剪辑预测。

        Parameters
        ----------
        x : tensor
            包含x_t的张量
        predicted_noise : tensor
            包含噪声的张量(无论是预测的还是真实的)
        t : tensor
            包含时间步长的张量
        x_0 : tensor
            包含x_0的张量

        Returns
        -------
        tensor
            包含x_t-1的张量
        """
        if predicted_noise == None and x_0 == None:
            print("Either noise or x_0 have to be given to calculate x_t-1.")
            exit(1)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
        if x_0 == None:
            pred_x0 = (
                1 / torch.sqrt(alpha_hat) * x
                - torch.sqrt((1 - alpha_hat) / (alpha_hat)) * predicted_noise
            )
            pred_x0 = pred_x0.clamp(-1, 1)
            x_0 = pred_x0
        w0 = torch.sqrt(alpha_hat_minus_one) * beta / (1 - alpha_hat)
        wt = torch.sqrt(alpha) * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
        return w0 * x_0 + wt * x

    def sample(
        self, model, n, channels, pyramid=False, simplex=False, discount = 0.8
    ):
        """这个方法从学习到的DDPM中采样。

            参数
            ----------
            型号：_type_
            学习后的DDPM模型（U-Net）
            N: int
            要创建的样本数量
            通道：int
            模型被训练的通道数。对于我来说，这是4 （FLAIR, T1, T1ce, T2）
            金字塔：bool，可选
            决定是否使用金字塔噪声的标志，默认情况下为False
            Simplex: bool，可选
            决定是否使用单形噪声的标志，默认情况下为False
            折扣：浮动，可选
            金字塔噪声的折扣，默认为0.8

            返回
            -------
            张量
            创建的样本。
        """
        model.eval()
        with torch.no_grad():
            if simplex == True:
                tmp = torch.randn((1, n * channels, self.img_size, self.img_size)).to(
                    self.device
                )
                slice_t = (torch.arange(1) + 10).long()
                x = baselines.simplex_noise.generate_simplex_noise(
                    tmp, slice_t, in_channels=tmp.shape[1]
                )
                x = x.view(n, x.shape[1] // n, self.img_size, self.img_size)
            elif pyramid == True:
                x = pyramid_noise_like(n, channels, self.img_size, discount, self.device)
            else:
                x = torch.randn((n, channels, self.img_size, self.img_size)).to(
                    self.device
                )
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                _, predicted_noise = model(x, t)
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex == True:
                        tmp = torch.randn(
                            (1, n * channels, self.img_size, self.img_size)
                        ).to(self.device)
                        slice_t = (torch.arange(1) + 10).long()
                        noise = baselines.simplex_noise.generate_simplex_noise(
                            tmp, slice_t, in_channels=tmp.shape[1]
                        )
                        noise = noise.view(
                            n, noise.shape[1] // n, noise.shape[2], noise.shape[3]
                        )
                    elif pyramid == True:
                        noise = pyramid_noise_like(n, channels, self.img_size, discount, x.device)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(var) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    def sample_context(
        self, model, n, channels, pyramid=False, simplex=False, discount=0.8, context=None
    ):
        """
        使用上下文条件从学习的扩散模型生成样本。

        参数:
        ----------
        model : nn.Module
            训练好的扩散模型
        n : int
            要生成的样本数量
        channels : int
            输入/输出通道数
        pyramid : bool
            是否使用金字塔噪声
        simplex : bool
            是否使用单纯形噪声
        discount : float
            金字塔噪声的折扣系数
        context : list
            来自MAE的上下文特征，格式为 [spatial_context, global_context]

        返回:
        -------
        tensor
            生成的样本图像，尺寸为 [n, channels, img_size, img_size]
        """
        model.eval()
        with torch.no_grad():
            # 初始化噪声
            if simplex:
                # 单纯形噪声
                tmp = torch.randn((1, n * channels, self.img_size, self.img_size)).to(
                    self.device
                )
                slice_t = (torch.arange(1) + 10).long()
                x = baselines.simplex_noise.generate_simplex_noise(
                    tmp, slice_t, in_channels=tmp.shape[1]
                )
                x = x.view(n, x.shape[1] // n, self.img_size, self.img_size)
            elif pyramid:
                # 金字塔噪声
                x = pyramid_noise_like(n, channels, self.img_size, discount, self.device)
            else:
                # 标准高斯噪声
                x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            # 逐步去噪生成样本
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # 预测噪声（使用上下文条件）
                _, predicted_noise = model(x, t, context=context)

                # 计算去噪步骤参数
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]

                # 添加噪声（步骤越靠近开始，添加的噪声越多）
                if i > 1:
                    if simplex:
                        # 生成单纯形噪声
                        tmp = torch.randn((1, n * channels, self.img_size, self.img_size)).to(self.device)
                        slice_t = (torch.arange(1) + 10).long()
                        noise = baselines.simplex_noise.generate_simplex_noise(
                            tmp, slice_t, in_channels=tmp.shape[1]
                        )
                        noise = noise.view(n, noise.shape[1] // n, noise.shape[2], noise.shape[3])
                    elif pyramid:
                        # 生成金字塔噪声
                        noise = pyramid_noise_like(n, channels, self.img_size, discount, x.device)
                    else:
                        # 生成标准高斯噪声
                        noise = torch.randn_like(x)
                else:
                    # 最后一步不添加噪声
                    noise = torch.zeros_like(x)

                # 计算方差
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)

                # 使用DDPM去噪公式更新样本
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(var) * noise

        # 将模型切换回训练模式
        model.train()

        # 将生成结果从[-1,1]映射到[0,1]再到[0,255]
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x
    
    
    def normative_diffusion(self, model, images, start=75, stop=200, pyramid=False, discount = 0.8):
        """This method calculates the deviations for each time step t in the interval T_l, T_u.
        In the ANDi paper this method corresponds to line 3 to 9 in the pseudo-code.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each time step in the dimension with the index 1
        """
        if stop == None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)
            """
                原始andi
            """
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                # print('images', images.shape)#[155, 4, 128, 128]
                x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                # print('x_t', x_t.shape)#[155, 4, 128, 128]
                # print(noise.shape)#[155, 4, 128, 128]
                _, predicted_noise = model(x_t, t)
                
                mu_theta = self.ddpm_mu_t(x_t, predicted_noise, t) #[155, 4, 128, 128]
                mu_q = self.ddpm_mu_t(x_t, noise, t) #[155, 4, 128, 128]
                d_t = (mu_q - mu_theta) ** 2
                dts[:, i - start] = d_t
                      
        return dts

    def analyze_latent_compactness(self, model, images, start=75, stop=200):
        """
        在推理的时间步范围内，提取健康区域的潜在特征 z_middle。
        为了量化紧凑性，我们通常选取推理区间的中点（如 t=137）或均匀采样。
        """
        model.eval()
        num_images = images.shape[0]

        # 我们选取推理区间的三个代表性时间步：开始、中间、结束
        sample_ts = [start, (start + stop) // 2, stop - 1]
        all_step_latents = []

        with torch.no_grad():
            for i in sample_ts:
                t = (torch.ones(num_images) * i).long().to(self.device)
                # 模拟推理时的加噪过程 (match images * 2 - 1)
                x_t, _ = self.noise_images(images, t, pyramid=True)

                # 提取 z_middle [B, 512, 8, 8]
                z_middle, _ = model(x_t, t)

                # 将特征拉平并归一化
                # [B, 512*8*8] = [B, 32768]
                z_flat = z_middle.view(num_images, -1).cpu().numpy()

                # L2 归一化是计算“紧凑性”的标准做法，因为它排除了模长的干扰，只关注特征模式
                z_flat = z_flat / (np.linalg.norm(z_flat, axis=1, keepdims=True) + 1e-8)
                all_step_latents.append(z_flat)

        # 返回形状为 [Num_Healthy_Slices * 3, 32768] 的特征矩阵
        return np.concatenate(all_step_latents, axis=0)
            

    def normative_diffusion_context(self, model, images, context=None, start=75, stop=200, pyramid=False, discount=0.8):
        """计算给定时间步范围内，使用上下文条件的扩散模型的偏差。
        这是ANDi论文中的关键算法的条件版本，支持MAE上下文特征。

        参数:
        ----------
        model : 已训练的条件扩散模型（UNet_Context）
        images : tensor, 形状 [B, C, H, W], 原始样本 x_0
        context : 字典 {'spatial': 空间特征, 'global': 全局特征} 或 None
                  如果为None，则模型无条件运行
        start : int, 间隔下界 T_l
        stop : int, 间隔上界 T_u
        pyramid : bool, 是否使用金字塔噪声
        discount : float, 金字塔噪声的折扣因子

        返回:
        -------
        tensor : 形状 [B, stop-start, C, H, W], 每个时间步的偏差
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:  # 起点不能是原始样本 x_0
            start = 1

        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (num_images, stop - start, images.shape[1], images.shape[2], images.shape[3])
            ).to(self.device)

            # 对每个时间步计算偏差
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)

                # 生成带噪声的图像
                x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)

                # 使用模型预测噪声（条件版本）
                _, predicted_noise = model(x_t, t, context=context)

                # 计算偏差（使用ANDi论文中的公式）
                mu_theta = self.ddpm_mu_t(x_t, predicted_noise, t)
                mu_q = self.ddpm_mu_t(x_t, noise, t)
                d_t = (mu_q - mu_theta) ** 2

                dts[:, i - start] = d_t

        return dts


        
    def normative_blocks(
        self, model, images, start=75, stop=200, skip=25, pyramid=False, discount = 0.8
    ):
        """实验方法。没有在我的论文中描述。
它计算扩散马尔可夫链块的偏差，并将方差设置为零。

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        skip : int, optional
            The number of deviations to skip before calculating the deviation for a block, by default 25
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each block in the dimension with the index 1
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    int(((stop - start) / skip)),
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            t = (torch.ones(num_images) * stop - 1).long().to(self.device)
            x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount = discount)
            correct_chain = x_t
            predicted_chain = x_t

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(predicted_chain, t)
                predicted_chain = self.ddpm_mu_t(predicted_chain, predicted_noise, t)
                correct_chain = self.ddpm_mean_t(correct_chain, t, x_0=images)
                if i % skip == 0 or i == 1:
                    d_t = (correct_chain - predicted_chain) ** 2
                    dts[:, int((i - start) / skip)] = d_t
                    t = (torch.ones(num_images) * i - 1).long().to(self.device)
                    x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                    predicted_chain = x_t
                    correct_chain = x_t
        return dts

    def deviations_noise(self, model, images, start=75, stop=200, pyramid=False, discount = 0.8):
        """This method calculates the deviations for each time step t in the interval T_l, T_u on the noise level.
        This provides similar results.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each time step in the dimension with the index 1
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                predicted_noise = model(x_t, t)
                d_t = (predicted_noise - noise) ** 2
                dts[:, i - start] = d_t
        return dts

    def ano_ddpm(self, model, images, num_steps, simplex=False, pyramid=False, discount = 0.8):
        """This method implements the lesion localization from the AnoDDPM paper.
        It is faster than the original implementation for the simplex noise by creating dependent noise along the batch dimension.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        num_steps : int
            The number of steps, the diffuion markov chain is used. Equal to T_u.
        simplex : bool, optional
            flag that decides if simplex noise is used, by default False
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The pseudo-healthy created.
        """
        model.eval()
        num_images = images.shape[0] #[155, 4, 128, 128]
        with torch.no_grad():
            t = (torch.ones(num_images) * num_steps).long().to(self.device)
            x, noise = self.noise_images(images, t, simplex=simplex, pyramid=pyramid, discount=discount)
            if simplex == True:
                slice_t = (torch.arange(1) + 10).long()
                complete_noise = torch.randn(
                    (1, t[0] * x.shape[1], self.img_size, self.img_size)
                )
                complete_noise = baselines.simplex_noise.generate_simplex_noise(
                    complete_noise, slice_t, in_channels=complete_noise.shape[1]
                )
                complete_noise = complete_noise.view(
                    t[0], x.shape[1], self.img_size, self.img_size
                ).to(self.device)
            for i in tqdm(reversed(range(1, num_steps)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                _, predicted_noise = model(x, t)
                # print(predicted_noise.shape)
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex == True:
                        noise = complete_noise[None, i].repeat(x.shape[0], 1, 1, 1)
                        noise = random_transform_vectorized(noise)
                    elif pyramid == True:
                        noise = pyramid_noise_like(x.shape[0], x.shape[1], self.img_size, discount, x.device)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mean_t(x, t, predicted_noise) + torch.sqrt(var) * noise
        return x

    def ano_cddpm(self, model, images, num_steps, context, simplex=False, pyramid=False, discount=0.8):
        """这个方法基于AnoDDPM论文实现了条件扩散模型的病变定位。
        它通过在批次维度上创建相关噪声，比原始实现更快处理单纯形噪声。

        参数
        ----------
        model : *type*
            学习的条件DDPM模型 (U-Net_Context)
        images : tensor
            样本 x_0
        num_steps : int
            扩散马尔可夫链使用的步数。等于 T_u。
        context : dict
            包含从MAE编码器提取的特征，格式为{'spatial': spatial_features, 'global': global_features}
        simplex : bool, optional
            决定是否使用单纯形噪声的标志，默认为False
        pyramid : bool, optional
            决定是否使用金字塔噪声的标志，默认为False
        discount : float, optional
            金字塔噪声的折扣率，默认为0.8

        返回
        -------
        tensor
            创建的伪健康图像。
        """
        model.eval()
        num_images = images.shape[0] #[155, 4, 128, 128]
        with torch.no_grad():
            t = (torch.ones(num_images) * num_steps).long().to(self.device)
            x, noise = self.noise_images(images, t, simplex=simplex, pyramid=pyramid, discount=discount)
            if simplex == True:
                slice_t = (torch.arange(1) + 10).long()
                complete_noise = torch.randn(
                    (1, t[0] * x.shape[1], self.img_size, self.img_size)
                )
                complete_noise = baselines.simplex_noise.generate_simplex_noise(
                    complete_noise, slice_t, in_channels=complete_noise.shape[1]
                )
                complete_noise = complete_noise.view(
                    t[0], x.shape[1], self.img_size, self.img_size
                ).to(self.device)
            for i in tqdm(reversed(range(1, num_steps)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                # 使用上下文信息作为条件输入
                _, predicted_noise = model(x, t, context=context)
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex == True:
                        noise = complete_noise[None, i].repeat(x.shape[0], 1, 1, 1)
                        noise = random_transform_vectorized(noise)
                    elif pyramid == True:
                        noise = pyramid_noise_like(x.shape[0], x.shape[1], self.img_size, discount, x.device)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mean_t(x, t, predicted_noise) + torch.sqrt(var) * noise
        return x

    def ano_cddpm_ensemble(self, model, images, num_steps, context, noise_levels=[250, 500, 750], simplex=False, pyramid=False, discount=0.8):
        """这个方法实现了条件扩散模型的集成版本的病变定位。
        它通过集成多个不同噪声级别的结果，提高检测性能。

        参数
        ----------
        model : *type*
            学习的条件DDPM模型 (U-Net_Context)
        images : tensor
            样本 x_0
        num_steps : int
            基础扩散步数 (未使用，保留参数兼容性)
        context : dict
            包含从MAE编码器提取的特征，格式为{'spatial': spatial_features, 'global': global_features}
        noise_levels : list, optional
            要集成的噪声级别列表，默认为[250, 500, 750]
        simplex : bool, optional
            决定是否使用单纯形噪声的标志，默认为False
        pyramid : bool, optional
            决定是否使用金字塔噪声的标志，默认为False
        discount : float, optional
            金字塔噪声的折扣率，默认为0.8

        返回
        -------
        tensor
            集成生成的伪健康图像。
        """
        model.eval()

        # 为每个噪声级别生成重建图像
        reconstructions = []
        for steps in noise_levels:
            # 使用相同的上下文信息，但不同的噪声级别
            pseudo_healthy = self.ano_cddpm(
                model=model, 
                images=images, 
                num_steps=steps, 
                context=context,
                simplex=simplex, 
                pyramid=pyramid, 
                discount=discount
            )
            reconstructions.append(pseudo_healthy)

        # 集成多个重建结果（取平均）
        ensemble_reconstruction = torch.stack(reconstructions).mean(dim=0)

        return ensemble_reconstruction    
