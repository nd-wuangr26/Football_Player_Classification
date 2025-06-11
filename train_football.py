from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--train-path", "-p", type=str, default="data/football_train")
    parser.add_argument("--val-path", "-v", type=str, default="data/football_val")
    parser.add_argument("--tensorboard-path", "-t", type=str, default="tensorboard_dir")
    parser.add_argument("--image_size", "-s", type=int, default=224)
    parser.add_argument("--num_epochs", "-n", type=int, default=100)
    parser.add_argument("--lr", "-l", type=float, default=0.001, help="Learing rate of model")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--resume_training", "-r", type=bool, default=False)

    args = parser.parse_args()
    return args

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.85, 1.1),
            shear=5,
        ),
        ToTensor()
    ])
    train_dataset = ImageFolder(root=args.train_path, transform=transform)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    val_dataset = ImageFolder(root=args.val_path, transform=ToTensor())
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )
    #model = MyCNN(num_class=len(train_dataset.categorises)).to(device)
    model = resnet18(weights=ResNet18_Weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=len(train_dataset.classes), bias=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(params=model.parameters(), lr=args.lr)
    if args.resume_training:
        checkpoint = torch.load("last.pt")
        model.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["accuracy"]
    else:
        start_epoch = 0
        best_accuracy = -1
    num_inter = len(train_data_loader)
    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    for epoch in range(start_epoch, args.num_epochs):
        #Training mode
        progess_bar = tqdm(train_data_loader)
        num_inter_per_epoch = len(train_data_loader)
        model.train()
        for inter, (images, labels) in enumerate(progess_bar):
            imagess = images.to(device)
            labels = labels.to(device)
            # Forward pass
            output = model(imagess)
            loss_value = criterion(output, labels)
            progess_bar.set_description("Epoch: {}/{}, Loss: {:0.4f}".format(epoch+1, args.num_epochs, loss_value.item()))
            writer.add_scalar("Train/Loss", loss_value, global_step=epoch*num_inter_per_epoch+inter)
            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        #Evaluation mode
        model.eval()
        losses = []
        all_predictions = []
        all_labels = []
        for inter, (images, labels) in enumerate(val_data_loader):
            imagess = images.to(device)
            labels = labels.to(device)
            # Forward pass
            with torch.inference_mode():
                output = model(imagess)
            loss_value = criterion(output, labels)
            losses.append(loss_value.item())
            predictions = torch.argmax(output, dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
        acc = accuracy_score(all_labels, all_predictions)
        loss = np.mean(losses)
        cm = confusion_matrix
        print("Epoch: {}/{}, Loss: {:0.4f}, Accuracy: {}".format(epoch + 1, args.num_epochs, loss, acc))
        writer.add_scalar("Val/Loss", loss, global_step=epoch)
        writer.add_scalar("Val/Accuracy", acc, global_step=epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "accuracy": best_accuracy,
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "last.pt")
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(checkpoint, "best.pt")


if __name__ == "__main__":
    args = get_args()
    train(args)