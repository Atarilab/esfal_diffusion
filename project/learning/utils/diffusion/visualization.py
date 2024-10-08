import torch
import matplotlib.pyplot as plt
import numpy as np

def get_coefs_and_variance(t, alpha_prod):
    """
    Returns coefficient used to generate samples
    in the reverse diffusion process at step t.
    The first coefficient scales the original sample, 
    The second scales the last prediction.
    The last value returned is the std of the reverse
    diffusion process. It scales the standard noise added in the
    reverse diffusion process. 
    """
    alpha_prod_t = alpha_prod[t]
    alpha_prod_t_prev = alpha_prod[t-1] if t > 0 else 1.
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
    sigma_square_t = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    
    return pred_original_sample_coeff.item(), current_sample_coeff.item(), sigma_square_t.item()

@torch.no_grad()
def compute_noise_schedule_values(alpha_cum_prod, return_numpy:bool=True):
    """
    Compute noise scheduler values from alpha cumulative products
    Returns:
        - log(SNR)
        - SNR(t-1) - SNR(t)
        - SNR(t-1) / SNR(t)
        - Coefficient original sample (x_0) in reverse process
        - Coefficient previous sample (x_t) in reverse process
        - Variance in reverse process
    """
    T = len(alpha_cum_prod)
    snr = alpha_cum_prod / (1 - alpha_cum_prod)
    log_snr = torch.log(alpha_cum_prod / (1 - alpha_cum_prod) + 1e-12)
    snr_diff = snr[:-1] - snr[1:]
    snr_div = snr[:-1] / snr[1:]

    coef_original = torch.empty_like(alpha_cum_prod)
    coef_previous = torch.empty_like(alpha_cum_prod)
    variance = torch.empty_like(alpha_cum_prod)

    for t in range(T):
        (coef_original[t],
         coef_previous[t],
         variance[t]) = get_coefs_and_variance(t, alpha_cum_prod)
        
    if return_numpy:
        return (log_snr.numpy(),
                snr_diff.numpy(),
                snr_div.numpy(),
                coef_original.numpy(),
                coef_previous.numpy(),
                variance.numpy())
    
    return (log_snr,
            snr_diff,
            snr_div,
            coef_original,
            coef_previous,
            variance)

@torch.no_grad()
def get_noise_schedule_figures(training_batch, model:torch.nn.Module):
    snr_model = model.noise_scheduler.snr_model
    device = next(snr_model.parameters()).device
    timesteps = torch.arange(snr_model.num_timesteps).to(device)

    alpha_cum_prod = snr_model.alpha_cum_prod(timesteps)

    (log_snr, 
    snr_diff,
    snr_div,
    coef_original,
    coef_previous,
    variance) = compute_noise_schedule_values(alpha_cum_prod)
    f = lambda x : np.log(x + 1.) + 1.

    fig_list = []

    fig, ax = plt.subplots(1)
    ax.plot(log_snr)
    ax.set_xlabel("Diffusion step t")
    ax.set_title("log(SNR)")
    fig_list.append(fig)

    fig, ax = plt.subplots(1)
    ax.plot(f(snr_diff), label="softplus(SNR(t-1) - SNR(t)) + 1.")
    ax.plot(f(snr_div), label="softplus(SNR(t-1) / SNR(t)) + 1.")
    ax.set_xlabel("Diffusion step t")
    ax.set_title("SNR variation")
    ax.legend(loc='best')
    fig_list.append(fig)

    fig, ax = plt.subplots(1)
    ax.plot(coef_original, label="Coeff x_0")
    ax.plot(coef_previous, label="Coeff x_t")
    ax.plot(variance, label="Variance")
    ax.set_xlabel("Diffusion step t")
    ax.set_title("Coefficient reverse diffusion process")
    ax.legend(loc='best')
    fig_list.append(fig)

    return fig_list