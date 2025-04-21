import numpy as np
import matplotlib.pyplot as plt

spec = np.load("data\\spectrogramms\\1.npy")

plt.figure(figsize=(50, 20))
plt.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(format="%+2.0f dB")
plt.title("Cпектрограмма")
plt.xlabel("Время")
plt.ylabel("Частота")
plt.tight_layout()
plt.show()
