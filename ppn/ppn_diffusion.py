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
    def ppn_loop(self, kspaces, sens, mask, model,
                progress=False, device="cpu", sampleType="PPN", mixpercent=0.0):

        sample_fn = {
                    'PPN':self.ppn, 'DDPM': self.ddpm, 'DDIM':self.ddim,
                    'real':self.ppn, 'complex': self.ppn_complex, 'multicoil':self.ppn_multicoil
                } [sampleType]
            
        self.isComplex = sampleType in ['complex', 'multicoil']
        
        print("Sampling type: ", sampleType)

        if sampleType == "multicoil":
            self.sens = sens.to(device)
            self.mixstepsize = int(self.num_timesteps * mixpercent) 
        else:
            self.sens = None
            self.mixstepsize = 0

        self.mask = mask.to(device)
        self.knowns = kspaces.to(device) * mask
        x = th.randn_like(self.knowns.real, device=device)
        # if self.isComplex:
        #     x = th.randn_like(self.knowns, device=device)
        # else:
            
    
        _indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(_indices)
        
        for i in indices:  
            x = sample_fn (
                model,
                x,
                i
            ) 
        return x, self.num_timesteps

    def A(self, x):
        return self.mask * fft2_m(self.sens * x)

    def A_H(self, x):  # Hermitian transpose
        return torch.sum(torch.conj(self.sens) * ifft2_m(x * self.mask), dim=1).unsqueeze(dim=1)

    def kaczmarz(self, x, y,lamb=1.0): #[1, 15, 320, 320])
        x = x + lamb * self.A_H(y - self.A(x)) # [1, 15, 320, 320]) + [1, 1, 320, 320])
        return x

    def ppn_multicoil(self, model, x, t):
        # multiple condition
        x = self.ppn(model, x, t)
        # combine condition
        if t % self.mixstepsize == 0:
            lamb = 1.0 # lamb_schedule.get_current_lambda(t)
            x = self.kaczmarz(x, self.knowns, self.sens, self.mask, lamb=lamb)

        return x

    def ppn_complex(self, model, x, t):
        pass

    def ppn(self, model, x, t): # 0: x_0, 1: x_t

        ts = th.tensor([t]  * x.shape[0], device=x.device)

        def projector(x):
            x_space = to_space(x)
            x_space = merge_known_with_mask(x_space, self.knowns, self.mask)
            return from_space(x_space)

        return self._ppn_sample(model, x, ts, projector=projector)

    def _ppn_sample(self, model, x, t, projector):
        x_0 = self.p_mean_variance(model, x, t)["pred_xstart"] # predictor
        x_0_hat = projector(x_0)
        x = self.noisor(x_0_hat, t, x_0)  # new
        return x

    def noisor(self, x_0_hat, t, x_0):
        return x_0_hat #th.sqrt(alpha_bar_prev) * x_0_hat + th.sqrt(1-alpha_bar_prev)* th.randn_like(x_0_hat)


    def ddpm(self, model, x, t):
        ts = th.tensor([t]  * x.shape[0], device=x.device)
        return self.p_sample(model, x, ts)['sample']
    
    def ddim(self, model, x, t):
        ts = th.tensor([t]  * x.shape[0], device=x.device)
        return self.ddim_sample(model, x, ts)['sample']

