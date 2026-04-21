import matplotlib.pyplot as plt

epochs = list(range(1, 31))
train_loss = [
    1.4306, 1.3947, 1.5593, 1.5647, 1.4121, 1.3089, 1.1937, 1.0440, 0.9821, 0.8562,
    0.7899, 0.8031, 0.6888, 0.8060, 0.6525, 0.8440, 0.6916, 0.6937, 0.6483, 0.6122,
    0.6845, 0.5678, 0.5658, 0.5447, 0.5191, 0.4134, 0.6096, 0.6526, 0.5312, 0.4686
]

val_loss = [
    1.1350, 1.1857, 1.1622, 1.0830, 1.0453, 0.9765, 0.9240, 0.9407, 0.8794, 0.9222,
    0.9528, 0.9382, 0.9440, 0.9402, 0.9040, 0.8988, 0.9236, 0.8817, 0.9282, 0.9512,
    0.9168, 0.9170, 0.9471, 0.9549, 0.9058, 0.9286, 0.9488, 0.9426, 0.9412, 0.9633
]

plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

plt.savefig('learning_curve3.pdf')
plt.show()