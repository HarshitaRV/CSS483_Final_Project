import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

overlay_dir = '/Users/acchuaccount/CSS483/all_overlays'
out_path = '/tmp/Fig_3_overlay_panel.png'

files = sorted([f for f in os.listdir(overlay_dir) if f.endswith('_overlay.png')])[:4]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

for i, ax in enumerate(axes):
    if i >= len(files):
        ax.axis('off')
        continue
    fn = files[i]
    img = plt.imread(os.path.join(overlay_dir, fn))
    ax.imshow(img)
    ax.set_title(fn.replace('_overlay.png', ''), fontsize=10)
    ax.axis('off')

plt.suptitle('Representative QC Overlays (red=mask, green=ROI)', fontsize=12)
plt.tight_layout()
plt.savefig(out_path, dpi=170, bbox_inches='tight')
print('Saved', out_path)
