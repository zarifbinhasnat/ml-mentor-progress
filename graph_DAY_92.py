"""
Day 92 -- Masked Autoencoders (MAE): structured vs. random patch masking
viewed in the frequency domain.

Ties the Signal Lab insight to the lesson: MAE masks patches *randomly*
(not on a regular grid) partly because a periodic sampling pattern in the
spatial domain produces periodic replicas (aliasing spikes) in the 2D
Fourier spectrum -- exactly the comb-sampling / aliasing argument from
classical DSP. A random mask spreads its spectral energy broadband, like
noise, with no coherent spikes for the decoder to exploit as a shortcut.

No downloads, no internet. Synthetic 64x64 "image" made of an 8x8 patch
grid (patch size = 8px), mask ratio = 0.75, seeded for reproducibility.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

IMG_SIZE = 64
PATCH = 8
GRID = IMG_SIZE // PATCH          # 8x8 = 64 patches
N_PATCHES = GRID * GRID
MASK_RATIO = 0.75
N_MASKED = int(N_PATCHES * MASK_RATIO)  # 48 masked, 16 visible


def patches_to_pixels(patch_mask_2d):
    """Nearest-neighbor upsample an 8x8 patch-level mask to 64x64 pixels."""
    return np.kron(patch_mask_2d, np.ones((PATCH, PATCH)))


# --- Structured mask: keep 1 patch out of every 2x2 block (periodic) ---
structured_patch_mask = np.zeros((GRID, GRID))
structured_patch_mask[0::2, 1::2] = 1  # masked=1 on a regular sub-lattice
structured_patch_mask[1::2, 0::2] = 1
structured_patch_mask[1::2, 1::2] = 1
# ratio check: 3 of every 4 patches masked -> matches MASK_RATIO = 0.75

# --- Random mask: uniform random subset without replacement, same ratio ---
flat_idx = np.arange(N_PATCHES)
masked_idx = np.random.choice(flat_idx, size=N_MASKED, replace=False)
random_patch_mask = np.zeros(N_PATCHES)
random_patch_mask[masked_idx] = 1
random_patch_mask = random_patch_mask.reshape(GRID, GRID)

structured_pixels = patches_to_pixels(structured_patch_mask)
random_pixels = patches_to_pixels(random_patch_mask)


def log_magnitude_spectrum(mask_img):
    f = np.fft.fftshift(np.fft.fft2(mask_img))
    mag = np.log1p(np.abs(f))
    return mag


structured_spec = log_magnitude_spectrum(structured_pixels)
random_spec = log_magnitude_spectrum(random_pixels)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(structured_pixels, cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("Structured (grid) mask\n75% masked, periodic")
axes[0, 0].axis("off")

axes[0, 1].imshow(structured_spec, cmap="magma")
axes[0, 1].set_title("FFT magnitude (log)\nsharp periodic spikes = aliasing")
axes[0, 1].axis("off")

axes[1, 0].imshow(random_pixels, cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title("Random mask (MAE-style)\n75% masked, i.i.d.")
axes[1, 0].axis("off")

axes[1, 1].imshow(random_spec, cmap="magma")
axes[1, 1].set_title("FFT magnitude (log)\nbroadband, no coherent spikes")
axes[1, 1].axis("off")

fig.suptitle(
    "Day 92: Regular-grid masking aliases in frequency space;\n"
    "random masking (what MAE uses) does not",
    fontsize=12,
)
fig.tight_layout()
plt.savefig("graph_DAY_92.png", dpi=120, bbox_inches="tight")
