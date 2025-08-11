import ROOT
import GUNDAM


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
