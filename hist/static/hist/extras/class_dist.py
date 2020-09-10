import matplotlib.pyplot as plt

# Histology Class Distributions
plot = plt.gca()
plt.bar(1, 78786, 1/1.2, color="orange", label='IDC (+)')
plt.bar(2, 198738, 1/1.2, color="purple", label='Non-IDC (-)')
plt.title("Class Distribution - IDC (+) & Non-IDC (-)")
plt.ylabel("Data Samples")
plot.axes.get_xaxis().set_visible(False)
plt.legend()
plt.show()