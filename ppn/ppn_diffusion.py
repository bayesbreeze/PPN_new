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
        super().__init__(use_timesteps, **kwargs)


    @th.no_grad()
    def ppn_loop(self, imgs, kspaces, sens, mask, model,
                progress=False, device="cpu", sampleType="PPN", mixpercent=0.0):

        sample_fn = {
                    'PPN':self.ppn, 'DDPM': self.ddpm, 'DDIM':self.ddim,
                    'real':self.ppn, 'complex': self.ppn_complex, 'multicoil':self.ppn_multicoil
                } [sampleType]
        
        print("Sampling type: ", sampleType)

        imgs = th.from_numpy(imgs).to(device)
        mask = mask.to(device)

        if sampleType == "multicoil":
            sens = th.from_numpy(sens).to(device)
            known = th.from_numpy(kspaces).to(device) 
            known = known * mask
            mixstepsize = int(self.num_timesteps * mixpercent) 
        else:
            known = to_space(imgs) * mask # in kspace
            mixstepsize = 0
            
        _indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(_indices)
        
        x = th.randn_like(imgs, device=device)
        for i in indices:  
            x = sample_fn (
                model,
                x,
                i,
                mask,
                known,
                sens,
                mixstepsize
            ) 
        return x, self.num_timesteps

    def ppn_multicoil(self, model, x, t, mask, known, sens, mixstepsize):
        # multiple condition

        # combine condition
        pass

    def ppn_complex(self, model, x, t, mask, known, sens=None, mixstepsize=0):
        pass

    def ppn(self, model, x, t, mask, known, sens=None, mixstepsize=0): # 0: x_0, 1: x_t

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

    def noisor(self, x_0_hat, t, x_0):
        return th.sqrt(alpha_bar_prev) * x_0_hat + th.sqrt(1-alpha_bar_prev)* th.randn_like(x_0_hat)


    def ddpm(self, model, x, t, mask, known, sens=None, mixstepsize=0):
        ts = th.tensor([t]  * x.shape[0], device=x.device)
        return self.p_sample(model, x, ts)['sample']
    
    def ddim(self, model, x, t, mask, known, sens=None, mixstepsize=0):
        ts = th.tensor([t]  * x.shape[0], device=x.device)
        return self.ddim_sample(model, x, ts)['sample']

