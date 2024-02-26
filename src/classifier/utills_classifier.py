import os
import torch
import matplotlib.pyplot as plt



class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, out_dir, name):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, os.path.join(out_dir, f'best_{name}.pth'))


def save_model(epochs, model, optimizer, criterion, out_dir, name):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, os.path.join(out_dir, f'{name}.pth'))


def save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir):
    def plot_and_save(data, label, filename):
        plt.figure(figsize=(10, 7))
        plt.plot(data['train'], color='tab:blue', linestyle='-', label=f'train {label}')
        plt.plot(data['validation'], color='tab:red', linestyle='-', label=f'validation {label}')
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f'{filename}.png'))

    accuracy_data = {'train': train_acc, 'validation': valid_acc}
    loss_data = {'train': train_loss, 'validation': valid_loss}

    plot_and_save(accuracy_data, 'accuracy', 'accuracy')
    plot_and_save(loss_data, 'loss', 'loss')
