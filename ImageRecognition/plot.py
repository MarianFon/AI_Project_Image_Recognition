import matplotlib.pyplot as plt

epochs = list(range(1, 26))
train_loss = [1.7155, 1.5030, 1.4242, 1.3849, 1.3523, 1.2510, 1.1808, 1.0927, 0.9952, 0.9541, 0.8988, 0.8455, 0.7607, 0.6861, 0.5981, 0.5692, 0.5552, 0.5063, 0.4696, 0.4388, 0.4071, 0.3723, 0.3551, 0.3344, 0.3191]
val_loss = [1.2829, 1.1912, 1.1588, 1.1363, 1.1377, 1.0579, 1.0176, 1.0532, 0.9735, 0.9657, 0.9441, 0.9215, 0.8849, 0.8872, 0.9004, 0.8980, 0.9095, 0.9090, 0.9184, 0.9423, 0.9250, 0.9345, 0.9268, 0.9204, 0.9172 ]

plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

plt.savefig('learning_curve2.pdf')
plt.show()