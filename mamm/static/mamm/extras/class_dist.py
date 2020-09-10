import matplotlib.pyplot as plt

# PBD Ranges Distribution
plot = plt.gca()
plt.rcParams["figure.figsize"] = (7,2)
plt.bar(1, 586, 1/1.2, color="teal", label='0-20%')
plt.bar(2, 700, 1/1.2, color="crimson", label='21-40%')
plt.bar(3, 450, 1/1.2, color="orange", label='41-60%')
plt.bar(4, 1000, 1/1.2, color="indigo", label='61-80%')
plt.bar(5, 850, 1/1.2, color="darkgreen", label='81-90%')
plt.title("Percent-Based Density Ranges Distribution")
plt.ylabel("Data Samples")
plot.axes.get_xaxis().set_visible(False)
plt.legend(loc=2)
plt.show()