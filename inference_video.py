from torchvision.models import resnet18, ResNet18_Weights
import cv2
import torch.nn as nn
import torch
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--image_size", "-s", type=int, default=224)
    parser.add_argument("--video_path", "-i", type=str, default="./test_video/1.mp4")
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
    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.image_size, args.image_size))
        image = np.transpose(image, (2, 0, 1))/255.
        image = image[None, :, :, :]
        image = torch.from_numpy(image).float().to(device) #Phai chuyen no ve tensor vi input cua model laf tensor
        # image = torch.unsqueeze(image, dim=0)   #Cach nay cung duoc
        softmax = nn.Softmax()
        with torch.no_grad():
            prediction = model(image)
            probs = softmax(prediction)[0]
            predicted_class = torch.argmax(probs)
            cv2.putText(frame, "{}: {:0.2f}%".format(categorises[predicted_class], probs[predicted_class]*100),
                        (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    args = get_args()
    inference(args)