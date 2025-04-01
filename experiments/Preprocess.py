import os
import cv2
from tqdm import tqdm

def preprocess_images(input_folder, output_folder, size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)
    for img_name in tqdm(os.listdir(input_folder)):
        img = cv2.imread(os.path.join(input_folder, img_name))
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_folder, img_name), img)

if __name__ == "__main__":
    preprocess_images("../datasets/monet2photo/trainA", "../datasets/monet2photo/trainA_resized")
    preprocess_images("../datasets/monet2photo/trainB", "../datasets/monet2photo/trainB_resized")
