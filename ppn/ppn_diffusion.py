from guided_diffusion.respace import *
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from .ppn_utils import *
import json

#"{'step_curve': '.5,.5,.5,.5', 'gamma': 0}"
class CommonParam:
    def __init__(self, json_str):
        self.data = json.loads(json_str)

    @property
    def step_curve(self):  # https://cubic-bezier.com/#.39,0,.66,.98
        if isinstance(self.data['step_curve'], str):
            self.data['step_curve']=tuple(float(i) for i in self.data['step_curve'].split(','))
        return self.data['step_curve']

    @property
    def gamma(self):  
        return self.data['gamma']

class PPN_Diffusion(SpacedDiffusion):

    def __init__(self, use_timesteps, **kwargs):
        # self.common_param = kwargs.pop('common_param')
        # self.gamma = self.common_param.gamma
        super().__init__(use_timesteps, **kwargs)


    @th.no_grad()
    def ppn_loop(self, model,test_imgs, 
            eta,   # -1: PPN, 0: DDIM, 1: DDPM
            proj,  # 0: x_0, 1: x_t
            acc,   # [4, 8, 16, 24]
            show_progress
        ):  
        assert eta in [-1, 0, 1], "invalid eta value: %s; shoule be -1: PPN, 0: DDIM, 1: DDPM" % eta
        assert proj in [0, 1], "invalid projection value: %s; should be 0 for x_0, 1 for x_t" % proj
        assert acc in [4, 8, 16, 24], "inavlid acceleration avlue: %s; shoule be [4, 8, 16]" % acc

        print(">> [Ablation] type: %s, proj: %s, acc: x%d; parameters: [eta: %d, proj: %d, acc: %d]" 
                % (["PPN", "DDIM", "DDPM"][eta+1], ["x0", "x1"][proj], acc, eta, proj, acc))

        shape = test_imgs.shape
        assert isinstance(shape, (tuple, list))
        device = next(model.parameters()).device
        img = th.randn(*shape, device=device)
        
        _indices = list(range(self.num_timesteps))[::-1]

        mask = get_cartesian_mask(shape[2:], int(shape[2]/acc)).to(device)
        known = to_space(test_imgs).to(device)

        if show_progress:
            from tqdm.auto import tqdm
            indices = tqdm(_indices)

        for i in indices:  
            img = self.ppn(
                model,
                img,
                i,
                mask,
                known,
                eta,
                proj
            ) 
        return img, self.num_timesteps

    def ppn(self,
        model, x, t, mask, known,
        eta, # -1: PPN, 0: DDIM, 1: DDPM
        proj): # 0: x_0, 1: x_t

        ts = th.tensor([t]  * x.shape[0], device=x.device)

        def projector(x):
            x_space = to_space(x)
            if proj==0: # project on x_0
                x_space = merge_known_with_mask(x_space, known, mask)
            else: # project on x_t
                alpha = _extract_into_tensor(self.sqrt_alphas_cumprod_prev, ts, x.shape)
                beta = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_prev, ts, x.shape)
                known_noisy = alpha * known + beta * to_space(th.rand_like(known))
                x_space = merge_known_with_mask(x_space, known_noisy, mask)

            return from_space(x_space)

        return self._ppn_sample(model, x, ts, proj=proj, projector=projector)

    def _ppn_sample(self, model, x, t, proj, projector):
        x_0 = self.p_mean_variance(model, x, t)["pred_xstart"] # predictor
        x_0_hat = projector(x_0)
        x = self.noisor2(x_0_hat, t, x_0)  # new
        return x

