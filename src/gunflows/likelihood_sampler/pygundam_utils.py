import ROOT
import GUNDAM
import numpy as np
import matplotlib.pyplot as plt

def convert_TH2D_to_TMatrix(th2d):
    if not isinstance(th2d, ROOT.TH2D):
        raise TypeError("Input must be a ROOT.TH2D object.")
    n_bins_x = th2d.GetNbinsX()
    n_bins_y = th2d.GetNbinsY()
    matrix = GUNDAM.TMatrixD(n_bins_x, n_bins_y)
    for i in range(1, n_bins_x + 1):
        for j in range(1, n_bins_y + 1):
            matrix[i - 1, j - 1] = th2d.GetBinContent(i, j)
    return matrix
    # return matrix


def big_vector_summary(vec, n_show=4):

    if len(vec) == 0:
        return "[]"
    if len(vec) > 10:
        if n_show < 1:
            raise ValueError("n_show must be at least 1.")
        if n_show > len(vec)/2 - 1:
            n_show = len(vec)
        return_string = "["
        for i in range(n_show):
            return_string += f"{vec[i]:.2f}, "
        return_string += "... "
        for i in range(-n_show, 0):
            return_string += f", {vec[i]:.2f} "
        return_string += f"] (length: {len(vec)})"
        return return_string
        # return f"[{vec[0]:.2f}, {vec[1]:.2f},{vec[2]:.2f}, {vec[3]:.2f}, ..., {vec[-4]:.2f}, {vec[-3]:.2f}, {vec[-2]:.2f}, {vec[-1]:.2f}] (length: {len(vec)})"
    else:
        return str(vec)

def log_multivariate_normal_pdf(x, mean, cov, with_log_det = False, precomputed_log_det=-1):
    eigen_parameter_values = convert_to_eigenspace(x, mean=mean, cov=cov)
    log_det = 0
    if with_log_det:
        if precomputed_log_det == -1:
            log_det = np.linalg.slogdet(cov)[1]  # log determinant
        else:
            log_det = precomputed_log_det

    log_prob = len(eigen_parameter_values)*0.5*(np.log(2*np.pi))+ 0.5*log_det + sum(eigen_parameter_values**2) # this is actually -log prob
    return log_prob

def convert_to_eigenspace(x, mean, cov):
    """
    Convert a vector x to the eigenspace of the covariance matrix.
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    # get cholesky decomposition of covariance matrix
    L = np.linalg.cholesky(cov)
    # eigen = L_inv @ (x - mean)
    L_inv = np.linalg.inv(L)
    eigen = L_inv @ (x - mean)
    return eigen



def draw_logp_logq(log_p, log_q, bestfit_nll, out_dir):

    # shift for visualization
    A = np.mean(log_p - bestfit_nll)
    B = np.mean(log_q)
    log_p = log_p + (B-A)

    plt.figure(figsize=(8, 6))
    plt.hist(log_p[log_p<5000] , alpha=0.7, density=True,
             color='lightblue', bins=100, label='-log p - bestfit_nll')
    plt.xlabel('NLL')
    plt.ylabel('Density')
    plt.title('NLL Distribution')
    plt.savefig(out_dir+'/NLL_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.hist(log_q, alpha=0.7, density=True,
             color='orange', bins=100,  label='-log q')
    mu, std = np.mean(log_q), np.std(log_q)
    if std == 0:
        std = 1e-6  # avoid division by zero in case of constant values
    # write mu and std on the canvas
    x = np.linspace(mu - 4*std, mu + 4*std, 1000)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    plt.plot(x, p, color='red', linewidth=2, label='Gaussian Fit')
    plt.plot([], [], ' ', label='$\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(mu, std))
    plt.legend()
    plt.xlabel('NLL')
    plt.ylabel('Density')
    plt.title('gNLL Distribution')
    plt.savefig(out_dir+'/gNLL_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    # 2d histogram of NLL vs gNLL
    h = plt.hist2d(log_p - bestfit_nll, log_q, bins=100, cmap='viridis', density=True, norm='log')
    plt.colorbar(h[3], label='Log Density')
    # diagonal line
    plt.plot([min(log_p - bestfit_nll), max(log_p - bestfit_nll)], [min(log_p - bestfit_nll), max(log_p - bestfit_nll)], color='red', linestyle='--')
    # adjust axes to be equal
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xlabel('NLL')
    plt.ylabel('gNLL')
    # print number of entries
    n_entries = len(log_p)
    plt.title(f'NLL vs gNLL Histogram ({n_entries} entries)')
    # save pics
    plt.savefig(out_dir+'/NLL_gNLL_histogram.png', dpi=100, bbox_inches='tight')
    plt.close()
    # NLL and gNLL overlaid
    plt.figure(figsize=(8, 6))
    plt.hist(log_q, alpha=0.7, density=True,
             color='lightblue', bins=100,  label='-log q')
    plt.hist(log_p - bestfit_nll, alpha=0.7, density=True,
             color='orange', bins=100, label='-log p - bestfit_nll')
    plt.xlabel('NLL')
    plt.ylabel('Density')
    plt.title('NLL and gNLL Distribution')
    # overlay line at best fit NLL
    plt.axvline(bestfit_nll, color='red', linestyle='--', label='Best Fit NLL')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save the plot
    plt.legend()
    plt.savefig(out_dir+'/NLL_gNLL_overlay.png', dpi=100, bbox_inches='tight')
    plt.close()

