from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False

IntOrPair = Union[int, Tuple[int, int]]

def _to_pair(x: IntOrPair) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else x

def _out_size_2d(HW_in, k, s, p, d):
    H, W = HW_in
    Ky, Kx = _to_pair(k)
    Sy, Sx = _to_pair(s)
    Py, Px = _to_pair(p)
    Dy, Dx = _to_pair(d)
    Hout = (H + 2*Py - Dy*(Ky-1) - 1) // Sy + 1
    Wout = (W + 2*Px - Dx*(Kx-1) - 1) // Sx + 1
    return Hout, Wout

class D2SConvS2D(nn.Module):
    """
    Depth-to-Space (PixelShuffle r) → Conv2d → Space-to-Depth (PixelUnshuffle r)
    - 目標：在「外部視角」模擬一個 Conv2d(k, stride, padding, dilation)，
      但中間透過 D2S/S2D 調整資料流以利硬體（或作 layer_path-version）。
    - 需求：in_ch % (r^2) == 0, out_ch % (r^2) == 0
    - 算量：與「D2S→Conv(K_out/r^2)→S2D」邏輯一致（若 out_ch 為原值，則屬降算量版）。
    - 形狀：本層會把輸出裁切為與「普通 Conv2d(k,stride,padding,dilation)」一致的 H×W。
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: IntOrPair = 1,
                 stride: IntOrPair = 1,
                 padding: IntOrPair = 0,
                 dilation: IntOrPair = 1,
                 r: int = 2,
                 groups: int = 1,
                 bias: bool = False):
        super().__init__()
        assert in_ch  % (r*r) == 0, "in_ch 必須可被 r^2 整除"
        assert out_ch % (r*r) == 0, "out_ch 必須可被 r^2 整除"

        self.r = r
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.d2s = nn.PixelShuffle(r)

        # 中：Conv。為了近似外部 padding 效果，內部 padding 先設為「外部 padding × r」。
        # 註：對 dilation>1 的嚴格等價要做額外推導；這裡優先保形狀（下面 forward 會再裁切）。
        Py, Px = _to_pair(padding)
        inner_padding = (Py * r, Px * r)

        self.mid_in  = in_ch // (r*r)
        self.mid_out = out_ch // (r*r)

        # DSCONV
        if groups > self.mid_in:
            groups = self.mid_in

        self.conv = nn.Conv2d(
            in_channels=self.mid_in,
            out_channels=self.mid_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=inner_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.s2d = nn.PixelUnshuffle(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        self.last_input_shape = x.shape
        H_tgt, W_tgt = _out_size_2d(
            (H, W),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        if DEBUG:
            print(f"Input shape: {x.shape}, Target output shape: {(N, self.out_ch, H_tgt, W_tgt)}")
            print(f"stride: {self.stride}, kernel size: {self.kernel_size}, padding: {self.padding}, dilation: {self.dilation}, r: {self.r}")

        # D2S → Conv
        y = self.d2s(x)        # (N, C/r^2, H*r, W*r)
        if DEBUG:
            print(f"real conv input shape: {y.shape}, kernel size: {self.conv.kernel_size}, ")
            print(f"input channels: {y.shape[1]}, output channels: {self.conv.out_channels}, ")
            print(f"stride: {self.conv.stride}, padding: {self.conv.padding}, dilation: {self.conv.dilation}")

        y = self.conv(y)       # (N, out/r^2, H', W') on the upscaled grid
        #if DEBUG:
        #    print(f"After Conv shape: {y.shape}")
        # 確保 H', W' 可被 r 整除（PixelUnshuffle 的需求）
        _, _, Hy, Wy = y.shape
        pad_h = (-Hy) % self.r
        pad_w = (-Wy) % self.r
        if pad_h or pad_w:
            # 只在下/右側補 0，避免位移
            y = F.pad(y, (0, pad_w, 0, pad_h))
        #if DEBUG:
        #    print(f"After padding shape: {y.shape}")
        # 再做 S2D
        y = self.s2d(y)        # (N, out, ?, ?)
        #if DEBUG:
        #    print(f"After S2D shape: {y.shape}")
        # 裁切到目標尺寸（與普通 Conv2d 對齊）
        y = y[:, :, :H_tgt, :W_tgt]
        return y

if __name__ == "__main__":
    test_nn = D2SConvS2D(in_ch=512, out_ch=512, kernel_size=3, stride=2, padding=0, r=2)
    print(test_nn)
    test_nn.eval()
    # Test the module with a dummy input
    dummy_input = torch.randn(1, 512, 14, 14)  # dummy input with shape (N, C, H, W)
    print(f"Input shape: {dummy_input.shape}")
    output = test_nn(dummy_input)
    print(f"Output shape: {output.shape}")