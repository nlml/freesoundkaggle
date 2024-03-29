import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi / (2 * 255.)
        self.eps = eps / (2 * 255.)
        self.ip = ip

    def adv_dist_metric(self, logp_hat, pred):
        # phat is logits, p is probabilities
        return F.binary_cross_entropy_with_logits(logp_hat, pred)
#         # MSE mode
#         return F.mse_loss(torch.sigmoid(logp_hat), pred)

    def forward(self, model, x, pred=None, return_adv=False):
        if pred is None:
            with torch.no_grad():
                pred = torch.sigmoid(model(x))

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                logp_hat = model(x + self.xi * d)
                adv_distance = self.adv_dist_metric(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            x_adv = x + r_adv
            logp_hat = model(x_adv)
            lds = self.adv_dist_metric(logp_hat, pred)

        if return_adv:
            return [i.cpu() for i in [lds, x_adv, torch.sigmoid(logp_hat), pred]]
        return lds
