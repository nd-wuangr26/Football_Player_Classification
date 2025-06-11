import os.path
import shutil
import json
import cv2
from glob import glob
from random import choices
from pprint import pprint
import numpy as np


def generate_dataset(data_path, output_path):
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    num_unknown_class = 1000
    for i in range(0, 11):
        os.makedirs(os.path.join(output_path, str(i)))
    for dir_path in os.listdir(data_path):
        video_path = os.path.join(data_path, dir_path, "{}.mp4".format(dir_path))
        json_path = os.path.join(data_path, dir_path, "{}.json".format(dir_path))
        cap = cv2.VideoCapture(video_path)
        with open(json_path, "r") as f:
            annotation = json.load(f)["annotations"]
        annotation = [anno for anno in annotation if anno["category_id"] == 4]
        # numbers = [anno["attributes"]["jersey_number"] for anno in annotation]
        # bbox = [anno["bbox"] for anno in annotation]
        # print(numbers, bbox)
        # input(len(annotation))
        while cap.isOpened():
            flag, frame = cap.read()
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not flag:
                break
            current_annotation = [anno for anno in annotation if anno["image_id"] == int(frame_id)]
            for anno in current_annotation:
                if anno["attributes"]["occluded"] != "no_occluded":
                    continue
                xmin, ymin, w, h = np.array(anno["bbox"], dtype=np.int32)
                jersey_number = anno["attributes"]["jersey_number"]
                team_jersey_color = anno["attributes"]["team_jersey_color"]
                jersey_visible = anno["attributes"]["number_visible"]
                player_image = frame[ymin:ymin+h, xmin:xmin+w, :]
                if jersey_visible != "visible":
                    jersey_number = "0"
                player_path = os.path.join(output_path, jersey_number,
                                         "{}_{}_{}_{}.jpg".format(dir_path, frame_id, team_jersey_color, jersey_number))
                cv2.imwrite(player_path, player_image)
    image_paths = [image_path for image_path in glob("{}/*.jpg".format(os.path.join(output_path, "0")))]
    select_images = choices(image_paths, k=num_unknown_class)
    for image_path in image_paths:
        if image_path not in select_images:
            os.remove(image_path)

if __name__ == "__main__":
    data_path = "dataset/football_test"
    output_path = "data/football_val"
    generate_dataset(data_path, output_path)