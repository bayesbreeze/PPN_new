from guided_diffusion.respace import *
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from .ppn_sample_utils import *
import json

# #"{'step_curve': '.5,.5,.5,.5', 'gamma': 0}"
# class CommonParam:
#     def __init__(self, json_str):
#         self.data = json.loads(json_str)

#     @property
#     def step_curve(self):  # https://cubic-bezier.com/#.39,0,.66,.98
#         if isinstance(self.data['step_curve'], str):
#             self.data['step_curve']=tuple(float(i) for i in self.data['step_curve'].split(','))
#         return self.data['step_curve']

#     @property
#     def gamma(self):  
#         return self.data['gamma']

class PPN_Diffusion(SpacedDiffusion):

    def __init__(self, use_timesteps, **kwargs):
        # self.common_param = kwargs.pop('common_param')
        # self.gamma = self.common_param.gamma
        super().__init__(use_timesteps, **kwargs)


    @th.no_grad()
    def ppn_loop(self, model, test_imgs, mask, progress=False, device="cpu"):  

        # print(">> [Ablation] type: %s, proj: %s, acc: x%d; parameters: [eta: %d, proj: %d, acc: %d]" 
        #         % (["PPN", "DDIM", "DDPM"][eta+1], ["x0", "x1"][proj], acc, eta, proj, acc))

        img = th.randn_like(test_imgs, device=device)
        test_imgs = test_imgs.to(device)
        known = to_space(test_imgs) # in kspace

        _indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(_indices)

        for i in indices:  
            img = self.ppn(
                model,
                img,
                i,
                mask,
                known
            ) 
        return img, self.num_timesteps

    def ppn(self, model, x, t, mask, known): # 0: x_0, 1: x_t

        ts = th.tensor([t]  * x.shape[0], device=x.device)

        def projector(x):
            x_space = to_space(x)
            x_space = merge_known_with_mask(x_space, known, mask)
            return from_space(x_space)

        return self._ppn_sample(model, x, ts, projector=projector)

    def _ppn_sample(self, model, x, t, projector):
        x_0 = self.p_mean_variance(model, x, t)["pred_xstart"] # predictor
        x_0_hat = projector(x_0)
        x = self.noisor(x_0_hat, t, x_0)  # new
        return x

    def noiser(self, x_0_hat, t, x_0):
        return th.sqrt(alpha_bar_prev) * x_0_hat + th.sqrt(1-alpha_bar_prev)* th.randn_like(x_0_hat)
