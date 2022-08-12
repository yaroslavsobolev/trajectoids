import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4,0.7))
plt.imshow([[1, 0]])
plt.axis('off')
plt.colorbar(aspect=10)
# plt.tight_layout()
fig.savefig('misc/figures/viridis_colorbar.png', dpi=500)
plt.show()