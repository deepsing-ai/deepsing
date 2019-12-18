import torch

# code from https://github.com/irasin/Pytorch_WCT


def whiten_and_color(content_feature, style_feature, alpha=1):
    """
    A WCT function can be used directly between encoder and decoder
    """
    cf = content_feature.squeeze(0)#.double()
    c, ch, cw = cf.shape
    cf = cf.reshape(c, -1)
    c_mean = torch.mean(cf, 1, keepdim=True)
    cf = cf - c_mean
    c_cov = torch.mm(cf, cf.t()).div(ch*cw - 1)
    c_u, c_e, c_v = torch.svd(c_cov)

    # if necessary, use k-th largest eig-value
    k_c = c
    for i in range(c):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = c_e[:k_c].pow(-0.5)

    w_step1 = torch.mm(c_v[:, :k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, :k_c].t()))
    whitened = torch.mm(w_step2, cf)

    sf = style_feature.squeeze(0)#.double()
    c, sh, sw = sf.shape
    sf = sf.reshape(c, -1)
    s_mean = torch.mean(sf, 1, keepdim=True)
    sf = sf - s_mean
    s_cov = torch.mm(sf, sf.t()).div(sh*sw -1)
    s_u, s_e, s_v = torch.svd(s_cov)

    # if necessary, use k-th largest eig-value
    k_s = c
    for i in range(c):
        if s_e[i] < 0.00001:
            k_s = i
            break
    s_d = s_e[:k_s].pow(0.5)
    c_step1 = torch.mm(s_v[:, :k_s], torch.diag(s_d))
    c_step2 = torch.mm(c_step1, s_v[:, :k_s].t())
    colored = torch.mm(c_step2, whitened) + s_mean

    colored_feature = colored.reshape(c, ch, cw).unsqueeze(0).float()

    colored_feature = alpha * colored_feature + (1 - alpha) * content_feature
    return colored_feature


# a = torch.randn(1, 64, 128, 128)
# b = torch.randn(1, 64, 124, 122)
#
# c = whiten_and_color(a, b)
# c.shape
