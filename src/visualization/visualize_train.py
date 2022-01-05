import matplotlib.pyplot as plt

def plot_loss(loss_train, loss_val, epochs):
    plt.plot(range(epochs+1), loss_train, label='Training loss')
    plt.plot(range(epochs+1), loss_val, label='Validation loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig('reports/figures/train/loss.png')
    plt.clf()