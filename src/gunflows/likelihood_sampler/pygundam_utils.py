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

def log_multivariate_normal_pdf(x, mean, cov, with_log_det, precomputed_log_det=-1):
    if with_log_det and precomputed_log_det == -1:
        raise ValueError("If with_log_det is True, precomputed_log_det can't be -1. Provide arguments to specify.")
    eigen_parameter_values = convert_to_eigenspace(x, mean=mean, cov=cov)
    log_det = 0
    if with_log_det:
        if precomputed_log_det == -1:
            log_det = np.linalg.slogdet(cov)[1]  # log determinant
        else:
            log_det = precomputed_log_det

    log_prob = 0.5*log_det + 0.5*len(eigen_parameter_values)*(np.log(2*np.pi)) + 0.5*sum(eigen_parameter_values**2) # this is actually -log prob
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
    print(f"mean of log_p - bestfit_nll: {A:.2f}, mean of log_q: {B:.2f}, bestfit_nll: {bestfit_nll:.2f}")

    plt.figure(figsize=(8, 6))
    plt.hist(log_p[log_p<5000] , alpha=0.7, density=True,
             color='lightblue', bins=100, label='-log p - bestfit_nll')
    plt.xlabel('NLL')
    plt.ylabel('Density')
    plt.title('NLL Distribution')
    mu, std = np.mean(log_p), np.std(log_p)
    if std == 0:
        std = 1e-6  # avoid division by zero in case of constant values
    # write mu and std on the canvas
    plt.axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.2f}')
    plt.axvline(mu + std, color='green', linestyle='--', label=f'Std: {std:.2f}')
    plt.axvline(mu - std, color='green', linestyle='--')
    plt.legend()
    plt.savefig(out_dir+'/NLL_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    log_p = log_p + (B-A)


    plt.hist(log_q, alpha=0.7, density=True,
             color='orange', bins=100,  label='-log q')
    mu, std = np.mean(log_q), np.std(log_q)
    if std == 0:
        std = 1e-6  # avoid division by zero in case of constant values
    # write mu and std on the canvas
    x = np.linspace(mu - 4*std, mu + 4*std, 1000)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    plt.plot(x, p, color='red', linewidth=2, label='Gaussian Fit')
    plt.plot([], [], ' ', label='$\\mu$ = {:.2f}, $\\sigma$ = {:.2f}'.format(mu, std))
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
    # Compute common bins for equal bin width when overlaying the two histograms
    arr_q = np.asarray(log_q)
    arr_p = np.asarray(log_p - bestfit_nll)
    # filter non-finite values
    arr_q = arr_q[np.isfinite(arr_q)]
    arr_p = arr_p[np.isfinite(arr_p)]
    # fall back if one of the arrays is empty
    if arr_q.size == 0 and arr_p.size == 0:
        common_bins = 100
    else:
        combined_min = np.min(arr_q) if arr_q.size > 0 else np.min(arr_p)
        combined_max = np.max(arr_q) if arr_q.size > 0 else np.max(arr_p)
        if arr_p.size > 0:
            combined_min = min(combined_min, np.min(arr_p))
            combined_max = max(combined_max, np.max(arr_p))
        # if range is zero (constant values), expand a bit
        if combined_max == combined_min:
            combined_min -= 0.5
            combined_max += 0.5
        n_bins = 100
        common_bins = np.linspace(combined_min, combined_max, n_bins + 1)

    plt.hist(arr_q, bins=common_bins, alpha=0.7, density=True,
             color='lightblue', label='-log q')
    plt.hist(arr_p, bins=common_bins, alpha=0.5, density=True,
             color='orange', label=f'-log p - bestfit_nll (shifted by {B-A:.1f})')
    plt.xlabel('NLL')
    plt.ylabel('Density')
    plt.title('NLL and gNLL Distribution')
    # overlay line at best fit NLL
    # plt.axvline(bestfit_nll, color='red', linestyle='--', label='Best Fit NLL')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save the plot
    plt.legend()
    plt.savefig(out_dir+'/NLL_gNLL_overlay.png', dpi=100, bbox_inches='tight')
    plt.close()

