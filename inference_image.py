from torchvision.models import resnet18, ResNet18_Weights
import cv2
import torch.nn as nn
import torch
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--image_size", "-s", type=int, default=224)
    parser.add_argument("--image_path", "-i", type=str, default="D:/Du_lieu_tong_hop/DeepLearning_VietNguyen/ex/ex4/test/test.jpg")
    parser.add_argument("--checkpoint", "-c", type=str, default="best.pt")

    args = parser.parse_args()
    return args

def inference(args):
    categorises = ["butterfly", "cat", "chicken", "cow", "dog",
                   "elephant", "horse", "sheep", "spider", "squirrel"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = resnet18(weights=ResNet18_Weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=10, bias=True)
    checkpoint = torch.load("best.pt")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float().to(device) #Phai chuyen no ve tensor vi input cua model laf tensor
    # image = torch.unsqueeze(image, dim=0)   #Cach nay cung duoc
    softmax = nn.Softmax()

    with torch.no_grad():
        prediction = model(image)
        print(prediction)
        probs = softmax(prediction)
        predicted_class = torch.argmax(probs)
        print(categorises[predicted_class])
        cv2.imshow(categorises[predicted_class], ori_image)
        cv2.waitKey(0)
        cv2.destroyWindow()

if __name__ == "__main__":
    args = get_args()
    inference(args)