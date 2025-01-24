import math
import torch
import torch.nn as nn


class SHConv(nn.Module):
    def __init__(self, in_ch, out_ch, L, interval):
        """
        The spectral convolutional filter has L+1 coefficients.
        Among the L+1 points, we set anchor points for every interval of "interval".
        Those anchors are linearly interpolated to fill the blank positions.

        Parameters
        __________
        in_ch : int
            # of input channels in this layer.
        out_ch : int
            # of output channels in this layer.
        L : int
            Bandwidth of input channels. An individual harmonic coefficient is learned in this bandwidth.
        interval : int
            Interval of anchor points. Harmonic coefficients are learned at every "interval".
            The intermediate coefficients between the anchor points are linearly interpolated.

        Notes
        _____
        Input shape  : [batch, in_ch, (L+1)**2]
        Output shape : [batch, out_ch, (L+1)**2]
        """

        super().__init__()

        ncpt = int(math.ceil(L / interval)) + 1 # upper bound integer of L / interval (ncpt: number of learnerable parameters)
        interval2 = 1 if interval == 1 else L - (ncpt - 2) * interval

        self.weight = nn.Parameter(torch.empty(in_ch, out_ch, ncpt, 1)) # if interval=5, L=80 -> self.weight only account for 16 values within the L's
        
        """
        [e.g. interval = 5, L = 80]
            ncpt = int(math.ceil(80 / 5)) + 1 = 17
            interval2 = 5
            
            l0, l1 -> coefficients for linear interpolation

            self.l0 =
            [[0.0, 0.2, 0.4, 0.6, 0.8],
             [0.0, 0.2, 0.4, 0.6, 0.8],
             ...,
             [0.0, 0.2, 0.4, 0.6, 0.8]], tensor_shape=(ncpt-2, interval)=(15, 5)
            
            self.l1 =
            [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]], tensor_shape=(1, interval2+1)=(1, 6)

            self.repeats = [1, 3, 5, 7, ..., 161]
            -> this is the number of degree in order (rotation in each frequency)
        """
        self.l0 = nn.Parameter(
            torch.arange(0, 1, 1.0 / interval).repeat(1, ncpt - 2).view((ncpt - 2, interval)), requires_grad=False
        )
        self.l1 = nn.Parameter(torch.arange(0, 1 + 1e-8, 1.0 / interval2).view((1, interval2 + 1)), requires_grad=False)
        self.repeats = nn.Parameter(torch.tensor([(2 * l + 1) for l in range(L + 1)]), requires_grad=False)

        stdv = 1.0 / math.sqrt(in_ch * (L + 1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        [dimension]
        w1 (flatten(-2)): (in_ch, out_ch, (ncpt-2) * interval)
        self.l0: (ncpt-2, interval)
        self.weight[:, :, :-2, :]: (in_ch, out_ch, ncpt-2, 1)

        w2 (flatten(-2)): (in_ch, out_ch, interval2)
        self.l1: (1, interval2)
        self.weight[:, :, -2:-1, :]: (in_ch, out_ch, 1, 1)
        """
        w1 = (
            torch.mul((1 - self.l0), self.weight[:, :, :-2, :]) + torch.mul(self.l0, self.weight[:, :, 1:-1, :])
        ).flatten(-2) # linear interpolation for the first ncpt number of intervals
        w2 = (
            torch.mul((1 - self.l1), self.weight[:, :, -2:-1, :]) + torch.mul(self.l1, self.weight[:, :, -1:, :])
        ).flatten(-2) # linear interpolation for the last interval
        
        """
        - (ncpt - 2) * interval + interval2 = L + 1 (number of order with max of L has total of L + 1 l values starting from 0 to L)
        - dimension of torch.cat([w1, w2], dim=2) = (in_ch, out_ch, L + 1)
        
        - torch.repeat_interleave -> repeats each element of the last dimension in torch.cat([w1, w2], dim=2) according to the values in self.repeats
        - for each degree l, the corresponding coefficient is repeated 2l + 1 times (same weight for the same order along all degree) - H^2C^2
        
        if torch.cat(one example of the last dimension(dim=2) with interval) was[0(learnable), 1, 2, 3, 4(learnable), 2, 0, -2, -4(learnable), 0, 4, 8, 12(learnable), ...]
        then w = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, ...(x7), 4, ...(x9), 2, ...(x11)] since the sum of 2l + 1 from l = 0 to L is (L + 1)^2
        ((H0, H1, H1, H1, H2, H2, H2, H2, H2, ...) -> easy form to calcuate with x)
        dimension of w: (in_ch, out_ch, (L + 1)^2)
        """
        w = torch.repeat_interleave(torch.cat([w1, w2], dim=2), self.repeats, dim=2)
        
        """
        - Filter (\hat{h}_l) is repeated 2l + 1 times to align with the ordering of m = -l, ..., l for each degree l
        - the order of coefficient in x: (l, m) = (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), ...
        - w.unsqueeze(0): (1, out_ch, in_ch, (L+1)^2), x.unsqueeze(2): [batch, in_ch, 1, (L+1)^2]
        - dimension of x: [batch, out_ch, (L+1)^2]
        """
        x = torch.mul(w.unsqueeze(0), x.unsqueeze(2)).sum(1)

        return x


class SHT(nn.Module):
    def __init__(self, L, Y_inv, area):
        """
        Spherical harmonic transform (SHT).
        Spherical signals are transformed into the spectral components.

        Parameters
        __________
        L : int
            Bandwidth of SHT. This should match L in SpectralConv.
        Y_inv : 2D array, shape = [n_vertex, (L+1)**2]
            Matrix form of harmonic basis.
        area : 1D array
            Area per vertex.

        Notes
        _____
        Input shape  : [batch, n_ch, n_vertex]
        Output shape : [batch, n_ch, (L+1)**2]
        """

        super().__init__()

        self.Y_inv = Y_inv[:, : (L + 1) ** 2]
        self.area = area

    def forward(self, x):
        x = torch.mul(self.area, x)
        x = torch.matmul(x, self.Y_inv)

        return x


class ISHT(nn.Module):
    def __init__(self, Y):
        """
        Inverse spherical harmonic transform (ISHT).
        Spherical signals are reconstructed from the spectral components.

        Parameters
        __________
        Y : 2D array, shape = [(L+1)**2, n_vertex]
            Matrix form of harmonic basis.

        Notes
        _____
        Input shape  : [batch, n_ch, (L+1)**2]
        Output shape : [batch, n_ch, n_vertex]
        """

        super().__init__()

        self.Y = Y

    def forward(self, x):
        x = torch.matmul(x, self.Y[: x.shape[-1], :])

        return x
