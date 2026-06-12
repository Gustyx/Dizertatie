"""generate_thesis_plots.py

Generates all thesis figures. Each image-based figure uses a different
dataset category so the thesis showcases a variety of visual content.

  Figure  │ Category used            │ Rationale
  ────────┼──────────────────────────┼────────────────────────────────────────
  Fig 1   │ (bar chart – no image)   │ per-category entropy from avg JSONs
  Fig 2   │ (bar chart – no image)   │ full-avg correlation values
  Fig 3   │ portrait_1               │ skin-tone peaks give dramatic before/after
  Fig 4   │ street_view_1            │ urban scene, visually interesting RGB
  Fig 5   │ architectural_plan_1     │ peaked plain histogram; Logistic/Henon fail
  Fig 6   │ medical_scan_1           │ very high plain correlation (great scatter)
  Fig 8   │ street_view_1            │ matches encrypted grid for consistency
  Fig 9   │ architectural_plan_1     │ Henon failure is most extreme here
  Fig 10  │ portrait_1               │ consistent with Fig 3; portrait depth
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ── Output folder ─────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent / 'thesis_figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ── Base paths ────────────────────────────────────────────────────────────────
IMAGES_BASE = Path(__file__).parents[1] / 'shared' / 'images'


def _resolve(folder, stem):
    """Return path for stem, trying .png then .tiff."""
    for ext in ('.png', '.tiff', '.tif'):
        p = folder / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f'No .png/.tiff found for {stem} in {folder}')


def _plain(cat, stem):
    """Load a plain image from the given category (auto-detects extension)."""
    path = _resolve(IMAGES_BASE / cat / 'plain', stem)
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def _enc(cat, alg_key, stem):
    """Load encrypted image (auto-detects extension)."""
    path = _resolve(IMAGES_BASE / cat / 'encrypted', f'{alg_key}_enc_{stem}')
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def _enc_1bit(cat, alg_key, stem):
    """Load 1-bit-modified encrypted image (auto-detects extension)."""
    path = _resolve(IMAGES_BASE / cat / 'encrypted_plus_one_bit',
                    f'{alg_key}_enc_{stem}')
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


# Algorithm keys and display labels (consistent order everywhere)
ALG_KEYS   = ['aes_ctr', 'aes_cbc', 'triple_des', 'chacha20',
               'logistic_map', 'henon_map']
ALG_LABELS = ['AES-CTR', 'AES-CBC', '3DES', 'ChaCha20',
              'Logistic Map', 'Henon Map']

# Colour palette — one colour per algorithm (same order)
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#795548']

n_algs = len(ALG_KEYS)

# =============================================================================
# FIGURE 1 — Shannon entropy per category and algorithm (bar chart)
# Data: per-category avg JSON values (6 categories including Spatial)
# =============================================================================

categories = ['Architectural\nPlan', 'CCTV\nFootage', 'Medical\nScan',
              'Portrait', 'Street\nView', 'Spatial']

# Values from per-category aggregated_analysis_results/*_avg.json
entropy_data = {
    'AES-CTR':      [7.9999, 7.9994, 7.9998, 7.9999, 7.9999, 7.9999],
    'AES-CBC':      [7.9995, 7.9982, 7.9992, 7.9998, 7.9995, 7.9999],
    '3DES':         [7.9999, 7.9994, 7.9998, 7.9999, 7.9999, 7.9998],
    'ChaCha20':     [7.9999, 7.9994, 7.9998, 7.9999, 7.9999, 7.9998],
    'Logistic Map': [7.6908, 7.9822, 7.9259, 7.9820, 7.9825, 7.9347],
    'Henon Map':    [5.6021, 7.9689, 7.8200, 7.9833, 7.9725, 7.8214],
}

fig1, ax1 = plt.subplots(figsize=(13, 5))

n_cats = len(categories)
bar_width = 0.12
x = np.arange(n_cats)

for i, (alg, values) in enumerate(entropy_data.items()):
    offset = (i - n_algs / 2 + 0.5) * bar_width
    ax1.bar(x + offset, values, bar_width,
            label=alg, color=COLORS[i], edgecolor='white', linewidth=0.5)

ax1.axhline(y=7.999, color='red', linestyle='--', linewidth=1.5,
            label='Threshold (7.999)', zorder=5)

ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10)
ax1.set_ylabel('Entropy (bits/pixel)', fontsize=11)
ax1.set_ylim(7.5, 8.05)
ax1.set_yticks([7.5, 7.8, 7.9, 7.95, 7.999])
ax1.yaxis.set_tick_params(labelsize=9)
ax1.legend(loc='lower right', fontsize=9, ncol=2)
ax1.grid(axis='y', linestyle=':', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'entropy_bar.png', dpi=300, bbox_inches='tight')
print(f'Saved entropy_bar -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 2 — Pixel correlation per direction (cross-category full-avg values)
# =============================================================================

directions = ['Horizontal', 'Vertical', 'Diagonal']

# Overall values from full_analysis_results/*_full_avg.json
corr_data = {
    'AES-CTR':      [0.002087, 0.002096, 0.002136],
    'AES-CBC':      [0.005565, 0.002093, 0.002078],
    '3DES':         [0.002069, 0.002060, 0.002098],
    'ChaCha20':     [0.002077, 0.002128, 0.002082],
    'Logistic Map': [0.030345, 0.001941, 0.002275],
    'Henon Map':    [0.158836, 0.706065, 0.155527],
}

fig2, ax2 = plt.subplots(figsize=(10, 5))

n_dirs = len(directions)
bar_width_c = 0.13
xc = np.arange(n_dirs)

for i, (alg, values) in enumerate(corr_data.items()):
    offset = (i - n_algs / 2 + 0.5) * bar_width_c
    ax2.bar(xc + offset, values, bar_width_c,
            label=alg, color=COLORS[i], edgecolor='white', linewidth=0.5)

ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5,
            label='Threshold (0.01)', zorder=5)

ax2.set_ylim(0.0, 0.031)
ax2.set_yticks([0.000, 0.002, 0.004, 0.006, 0.01, 0.03])
ax2.yaxis.set_tick_params(labelsize=9)
ax2.set_ylabel('Absolute Pearson $r$', fontsize=11)
ax2.set_xticks(xc)
ax2.set_xticklabels(directions, fontsize=11)
ax2.legend(loc='upper right', fontsize=9, ncol=2)
ax2.grid(axis='y', linestyle=':', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Annotate Henon bars that exceed the axis with their actual value
for i, (alg, values) in enumerate(corr_data.items()):
    if alg == 'Henon Map':
        for j, v in enumerate(values):
            if v > 0.012:
                offset = (i - n_algs / 2 + 0.5) * bar_width_c
                ax2.annotate(f'{v:.3f}^', xy=(xc[j] + offset, 0.012),
                             ha='center', va='bottom', fontsize=7,
                             color=COLORS[i], fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'correlation_bars.png', dpi=300, bbox_inches='tight')
print(f'Saved correlation_bars -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 3 — Intensity histograms: plain vs AES-CTR (street_view_1)
# =============================================================================

plain_street    = _plain('street_view', 'street_view_3')
enc_street_ctr  = _enc('street_view', 'aes_ctr', 'street_view_3')

ch_colors = ['#E53935', '#43A047', '#1E88E5']
ch_names  = ['R', 'G', 'B']

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

for ax, img, title in zip(axes3,
                           [plain_street, enc_street_ctr],
                           ['Original image (street view)', 'AES-CTR encrypted']):
    for c, (col, name) in enumerate(zip(ch_colors, ch_names)):
        hist, bins = np.histogram(img[..., c].ravel(), bins=256, range=(0, 255))
        ax.plot(bins[:-1], hist, color=col, alpha=0.75, linewidth=0.8, label=name)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Pixel intensity', fontsize=10)
    ax.set_ylabel('Pixel count', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'hist_comparison.png', dpi=300, bbox_inches='tight')
print(f'Saved hist_comparison -> {FIGURES_DIR}')
plt.close()

# Re-load street view for later figures that need it
plain_street = _plain('street_view', 'street_view_1')

# =============================================================================
# FIGURE 4 — Encrypted image grid: plain + all 6 algorithms (street_view_1)
# =============================================================================

plain_street = _plain('cctv_footage', 'cctv_footage_1')
enc_street   = [_enc('cctv_footage', k, 'cctv_footage_1') for k in ALG_KEYS]

fig4, axes4 = plt.subplots(1, 7, figsize=(18, 4))
for ax, img, title in zip(axes4,
                           [plain_street] + enc_street,
                           ['Original'] + ALG_LABELS):
    ax.imshow(img)
    ax.set_title(title, fontsize=8, pad=4)
    ax.axis('off')

plt.tight_layout(pad=0.5)
plt.savefig(FIGURES_DIR / 'encrypted_grid.png', dpi=300, bbox_inches='tight')
print(f'Saved encrypted_grid -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 5 — Histograms for all 6 encrypted algorithms (architectural_plan_1)
# =============================================================================

enc_medical_all = [_enc('medical_scan', k, 'medical_scan_1') for k in ALG_KEYS]
enc_arch        = [_enc('architectural_plan', k, 'architectural_plan_1') for k in ALG_KEYS]

fig5, axes5 = plt.subplots(2, 3, figsize=(13, 6), sharey=False)

for ax, img, label in zip(axes5.flat, enc_medical_all, ALG_LABELS):
    hist, bins = np.histogram(img[..., 0].ravel(), bins=256, range=(0, 255))
    ax.plot(bins[:-1], hist, color='#E53935', linewidth=0.8, alpha=0.85)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Pixel intensity', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.grid(linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Encrypted image histograms - R channel (medical scan)', fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'hist_all_algorithms.png', dpi=300, bbox_inches='tight')
print(f'Saved hist_all_algorithms -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 6 — Correlation scatter: Original / AES-CTR / Henon (medical_scan_1)
# Medical scans have very high spatial correlation in the plain image.
# =============================================================================

N_SAMPLES = 5000
rng = np.random.default_rng(42)


def scatter_pairs(img_channel, direction, n=N_SAMPLES):
    """Return (x, y) arrays of n sampled adjacent-pixel pairs."""
    h, w = img_channel.shape
    if direction == 'H':
        xs = rng.integers(0, w - 1, n)
        ys = rng.integers(0, h,     n)
        return img_channel[ys, xs], img_channel[ys, xs + 1]
    elif direction == 'V':
        xs = rng.integers(0, w,     n)
        ys = rng.integers(0, h - 1, n)
        return img_channel[ys, xs], img_channel[ys + 1, xs]
    else:  # diagonal
        xs = rng.integers(0, w - 1, n)
        ys = rng.integers(0, h - 1, n)
        return img_channel[ys, xs], img_channel[ys + 1, xs + 1]


plain_spatial     = _plain('spatial', 'spatial_1')
enc_spatial_ctr   = _enc('spatial', 'aes_ctr',    'spatial_1')
enc_spatial_henon = _enc('spatial', 'henon_map',  'spatial_1')

dirs       = ['H', 'V', 'D']
dir_labels = ['Horizontal', 'Vertical', 'Diagonal']

BG_COLOR = '#1a1a2e'   # dark navy background for all scatter subplots

row_configs = [
    (plain_spatial,     'Original',   '#00E5FF'),  # vivid cyan
    (enc_spatial_ctr,   'AES-CTR',    '#69FF47'),  # vivid lime green
    (enc_spatial_henon, 'Henon Map',  '#FF6D00'),  # vivid deep orange
]

fig6, axes6 = plt.subplots(3, 3, figsize=(12, 10), facecolor=BG_COLOR)
fig6.subplots_adjust(hspace=0.45, wspace=0.35)

for row, (img, row_label, dot_color) in enumerate(row_configs):
    for col, (d, dl) in enumerate(zip(dirs, dir_labels)):
        ax = axes6[row, col]
        ax.set_facecolor(BG_COLOR)
        x_vals, y_vals = scatter_pairs(img[..., 0], d)
        ax.scatter(x_vals, y_vals, s=1.2, alpha=0.55, color=dot_color, linewidths=0)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_xlabel('Pixel $x_i$', fontsize=8, color='#cccccc')
        ax.set_ylabel('Pixel $x_{i+1}$', fontsize=8, color='#cccccc')
        ax.set_title(f'{row_label} -- {dl}', fontsize=9, color=dot_color, fontweight='bold')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

plt.savefig(FIGURES_DIR / 'scatter_comparison.png', dpi=200, bbox_inches='tight',
            facecolor=BG_COLOR)
print(f'Saved scatter_comparison -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 8 — Bit-plane analysis: plain vs AES-CTR (street_view_1)
# =============================================================================


def bit_planes(img_channel):
    """Return list of 8 bit-plane images (bit 7 = MSB first)."""
    return [((img_channel >> b) & 1).astype(np.uint8) * 255
            for b in range(7, -1, -1)]

plain_portrait = _plain('portrait', 'portrait_1')
enc_portrait_ctr_bp = _enc('portrait', 'aes_ctr', 'portrait_1')
plain_bp_ch         = np.dot(plain_portrait[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
enc_bp_ctr_ch       = np.dot(enc_portrait_ctr_bp[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

fig8, axes8 = plt.subplots(2, 8, figsize=(16, 4))
for row, (ch, row_label) in enumerate([(plain_bp_ch,   'Original'),
                                        (enc_bp_ctr_ch, 'AES-CTR')]):
    for col, (plane, bit) in enumerate(zip(bit_planes(ch), range(7, -1, -1))):
        ax = axes8[row, col]
        ax.imshow(plane, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        if row == 0:
            ax.set_title(f'Bit {bit}', fontsize=8)
    axes8[row, 0].set_ylabel(row_label, fontsize=9, rotation=90,
                              labelpad=4, va='center')

plt.tight_layout(pad=0.3)
plt.savefig(FIGURES_DIR / 'bitplane_comparison.png', dpi=200, bbox_inches='tight')
print(f'Saved bitplane_comparison -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 9 — Bit-plane analysis: plain vs Henon Map (architectural_plan_1)
# Architectural plans have large uniform regions — Henon's failure (structured
# patterns surviving in high-order planes) is most visible on this content.
# =============================================================================

enc_portrait_henon_bp = _enc('portrait', 'henon_map', 'portrait_1')
plain_arch_ch         = np.dot(plain_portrait[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
enc_arch_henon_ch     = np.dot(enc_portrait_henon_bp[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

fig9, axes9 = plt.subplots(2, 8, figsize=(16, 4))
for row, (ch, row_label) in enumerate([(plain_arch_ch,        'Original'),
                                        (enc_arch_henon_ch,    'Henon Map')]):
    for col, (plane, bit) in enumerate(zip(bit_planes(ch), range(7, -1, -1))):
        ax = axes9[row, col]
        ax.imshow(plane, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        if row == 0:
            ax.set_title(f'Bit {bit}', fontsize=8)
    axes9[row, 0].set_ylabel(row_label, fontsize=9, rotation=90,
                              labelpad=4, va='center')

plt.tight_layout(pad=0.3)
plt.savefig(FIGURES_DIR / 'bitplane_henon.png', dpi=200, bbox_inches='tight')
print(f'Saved bitplane_henon -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 10 — Differential difference images (portrait_1)
# =============================================================================

fig10, axes10 = plt.subplots(2, 3, figsize=(13, 7))
for ax, alg_key, label in zip(axes10.flat, ALG_KEYS, ALG_LABELS):
    enc1     = _enc('architectural_plan',     alg_key, 'architectural_plan_1')
    enc2     = _enc_1bit('architectural_plan', alg_key, 'architectural_plan_1')
    diff     = np.abs(enc1.astype(np.int16) - enc2.astype(np.int16))
    diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
    ax.imshow(diff_vis)
    ax.set_title(label, fontsize=10)
    ax.axis('off')

plt.suptitle(
    r'Amplified pixel difference: $\mathrm{Enc}(P)$ vs $\mathrm{Enc}(P \oplus 1\,\mathrm{bit})$',
    fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'diff_images.png', dpi=200, bbox_inches='tight')
print(f'Saved diff_images -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 11 — Encryption time per algorithm and category (log scale)
# =============================================================================

time_categories = ['Arch.\nPlan', 'CCTV', 'Medical', 'Portrait', 'Street', 'Spatial']

# Per-category average encryption times (seconds) from aggregated_analysis_results
time_data = {
    'AES-CTR':      [0.048, 0.082, 0.031, 0.120, 0.046, 0.432],
    'AES-CBC':      [0.050, 0.087, 0.034, 0.124, 0.048, 0.485],
    '3DES':         [0.289, 0.564, 0.186, 0.686, 0.284, 11.713],
    'ChaCha20':     [0.049, 0.085, 0.033, 0.121, 0.048, 0.422],
    'Logistic Map': [0.552, 1.246, 0.364, 1.379, 0.554, 31.368],
    'Henon Map':    [0.283, 0.584, 0.200, 0.711, 0.302, 15.071],
}

fig11, ax11 = plt.subplots(figsize=(13, 5))
n_tcat = len(time_categories)
bw = 0.13
x11 = np.arange(n_tcat)

for i, (alg, vals) in enumerate(time_data.items()):
    offset = (i - n_algs / 2 + 0.5) * bw
    ax11.bar(x11 + offset, vals, bw,
             label=alg, color=COLORS[i], edgecolor='white', linewidth=0.5)

ax11.set_yscale('log')
ax11.set_xticks(x11)
ax11.set_xticklabels(time_categories, fontsize=10)
ax11.set_ylabel('Encryption time (s) -- log scale', fontsize=11)
ax11.legend(fontsize=9, ncol=2)
ax11.grid(axis='y', linestyle=':', alpha=0.5)
ax11.spines['top'].set_visible(False)
ax11.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'enc_time_bar.png', dpi=300, bbox_inches='tight')
print(f'Saved enc_time_bar -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 12 — Cross-dataset entropy heat map
# Data: per-category avg JSON values (corrected spatial column)
# =============================================================================

entropy_matrix = np.array([
    # Arch,   CCTV,   Medical, Portrait, Street,  Spatial
    [7.9999, 7.9994, 7.9998,  7.9999,   7.9999,  7.9999],  # AES-CTR
    [7.9995, 7.9982, 7.9992,  7.9998,   7.9995,  7.9999],  # AES-CBC
    [7.9999, 7.9994, 7.9998,  7.9999,   7.9999,  7.9998],  # 3DES
    [7.9999, 7.9994, 7.9998,  7.9999,   7.9999,  7.9998],  # ChaCha20
    [7.6908, 7.9822, 7.9259,  7.9820,   7.9825,  7.9347],  # Logistic Map
    [5.6021, 7.9689, 7.8200,  7.9833,   7.9725,  7.8214],  # Henon Map
])

cat_labels_hm = ['Arch.\nPlan', 'CCTV', 'Medical', 'Portrait',
                 'Street\nView', 'Spatial']

fig12, ax12 = plt.subplots(figsize=(10, 5))
im = ax12.imshow(entropy_matrix, cmap='RdYlGn', vmin=5.0, vmax=8.0, aspect='auto')

ax12.set_xticks(range(len(cat_labels_hm)))
ax12.set_xticklabels(cat_labels_hm, fontsize=10)
ax12.set_yticks(range(len(ALG_LABELS)))
ax12.set_yticklabels(ALG_LABELS, fontsize=10)

for r in range(entropy_matrix.shape[0]):
    for c in range(entropy_matrix.shape[1]):
        val = entropy_matrix[r, c]
        text_color = 'black' if val > 6.5 else 'white'
        ax12.text(c, r, f'{val:.4f}', ha='center', va='center',
                  fontsize=8, color=text_color)

cbar = fig12.colorbar(im, ax=ax12, fraction=0.03, pad=0.02)
cbar.set_label('Entropy (bits/pixel)', fontsize=10)
ax12.set_title('Shannon entropy by algorithm and dataset category', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'heatmap_consistency.png', dpi=300, bbox_inches='tight')
print(f'Saved heatmap_consistency -> {FIGURES_DIR}')
plt.close()

# =============================================================================
# FIGURE 13 — Radar charts: 6 algorithms across 5 normalised dimensions
# Data: cross-category full-avg from full_analysis_results/*_full_avg.json
# =============================================================================

radar_dims = ['Entropy', 'Correlation\nResistance', 'Histogram\nUniformity',
              'SSIM\n(inverted)', 'Speed\n(inverted)']


def norm_entropy(v):    return max(0.0, (v - 5.0) / 3.0)
def norm_corr(v):       return max(0.0, 1.0 - v)
def norm_hist(v):       return min(1.0, v / 0.05) if v < 0.05 else 1.0
def norm_ssim(v):       return max(0.0, 1.0 - v / 0.05)
def norm_speed(t, mx):
    return max(0.0, 1.0 - np.log(t + 1e-9) / np.log(mx + 1e-9))


# Raw values from full_analysis_results/*_full_avg.json
#   entropy  : overall Shannon entropy
#   corr     : max(|H|, |V|, |D|)  (worst correlation direction)
#   hist     : chi2_p overall
#   ssim     : SSIM overall
#   speed    : encryption_time_seconds (cross-category avg)
max_time = 5.8461  # Logistic Map is the slowest

radar_raw = {
    #               entropy  corr      hist    ssim    speed
    'AES-CTR':     [7.9998,  0.002136, 0.2545, 0.0141, 0.1141],
    'AES-CBC':     [7.9993,  0.005565, 0.0207, 0.0141, 0.1287],
    '3DES':        [7.9998,  0.002098, 0.2501, 0.0141, 1.9675],
    'ChaCha20':    [7.9998,  0.002128, 0.2483, 0.0141, 0.1234],
    'Logistic Map':[7.9164,  0.030345, 0.0000, 0.0153, 5.8461],
    'Henon Map':   [7.5280,  0.706065, 0.0000, 0.0242, 2.2489],
}

radar_scores = {}
for alg, (ent, corr, hist, ssim, spd) in radar_raw.items():
    radar_scores[alg] = [
        norm_entropy(ent),
        norm_corr(corr),
        norm_hist(hist),
        norm_ssim(ssim),
        norm_speed(spd, max_time),
    ]

n_dims = len(radar_dims)
angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
angles += angles[:1]

fig13, axes13 = plt.subplots(2, 3, figsize=(14, 9),
                              subplot_kw=dict(polar=True))

for ax, (alg, scores), col in zip(axes13.flat, radar_scores.items(), COLORS):
    vals = scores + scores[:1]
    ax.plot(angles, vals, color=col, linewidth=2)
    ax.fill(angles, vals, color=col, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_dims, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=6)
    ax.set_title(alg, fontsize=10, pad=12, color=col, fontweight='bold')
    ax.grid(linestyle=':', alpha=0.5)

plt.tight_layout(pad=2.0)
plt.savefig(FIGURES_DIR / 'radar_comparison.png', dpi=300, bbox_inches='tight')
print(f'Saved radar_comparison -> {FIGURES_DIR}')
plt.close()

print(f'\nAll figures complete. Saved to {FIGURES_DIR}')
