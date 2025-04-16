import os
import cv2
import numpy as np
from sklearn.metrics import f1_score
from abc import *
import requests
import zipfile
import os

input_folder = "./exercise_input"
output_folder = "./exercise_output"

def download_dataset():
    zip_url = "https://github.com/BrawnyClover/DIP_WINTER/raw/main/exercise_input.zip"
    zip_path = "exercise_input.zip"
    response = requests.get(zip_url, stream=True)
    
    download_success = False
    if response.status_code == 200:
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Download Success: {zip_path}")
        download_success = True
        
    else:
        print(f"Download Failed... (HTTP {response.status_code})")

    if download_success is True:
        try:
            extract_folder = input_folder
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            print(f"Decompression complete!: {extract_folder}")
            os.remove(zip_path)
            print(f"ZIP file deletion completed: {zip_path}")

        except zipfile.BadZipFile:
            print("Decompression failed: The ZIP file may be corrupted.")

        
def load_images_dict(input_folder):
    images_dict = {}
    for filename in os.listdir(input_folder):
        if "_map" not in filename:  
            name, ext = os.path.splitext(filename)
            filepath = os.path.join(input_folder, filename)
            img = cv2.imread(filepath)
            if img is not None:
                images_dict[name] = img
    return images_dict

def load_ground_truth_dict(input_folder):
    gt_dict = {}
    for filename in os.listdir(input_folder):
        if "_map" in filename: 
            name, ext = os.path.splitext(filename)
            key_name = name.replace("_map", "").replace("_road","")
            
            filepath = os.path.join(input_folder, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                gt_dict[key_name] = binary_mask
    return gt_dict

def calculate_average_f1_score(predicted_dict, gt_dict):
    f1_scores = []
    common_keys = set(predicted_dict.keys()).intersection(set(gt_dict.keys()))

    for key in common_keys:
        pred_mask = predicted_dict[key]
        gt_mask   = gt_dict[key]

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        valid_mask = (gt_mask != 127)

        pred_flat = (pred_mask.flatten() > 127).astype(np.uint8)
        gt_flat   = (gt_mask.flatten() > 127).astype(np.uint8)
        valid_flat = valid_mask.flatten()

        if np.any(valid_flat):  
            score = f1_score(gt_flat[valid_flat], pred_flat[valid_flat])
            f1_scores.append(score)

    if len(f1_scores) == 0:
        return 0.0
    return np.mean(f1_scores)

if not os.path.exists(input_folder):
    download_dataset()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

images_dict = load_images_dict(input_folder)
print(f"Test set : {len(images_dict)} images")

gt_dict = load_ground_truth_dict(input_folder)
print(f"Segmentation mask : {len(gt_dict)} labels")


class SegmentationMapGenerator(metaclass=ABCMeta):
    
    @abstractmethod
    def predict_segmentation_map(self):
        pass

    @abstractmethod
    def predict_segmentation_maps(self):
        pass

#=========================<Your code starts here>========================

class RoadSegmentationMapGenerator(SegmentationMapGenerator):
    def __init__(self, images, out_folder):
        self.images = images
        self.out_folder = out_folder

    def predict_segmentation_map(self, img):
        # BGR to HSV 변환
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

        # 도로 색상 범위 지정 (회색 계열, 약간의 파란색 포함)
        lower_gray = np.array([0, 0, 50], dtype=np.uint8)
        upper_gray = np.array([180, 50, 200], dtype=np.uint8)
        road_mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Morphological 연산 적용 (노이즈 제거 및 경계 부드럽게)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        pred_mask = cv2.dilate(road_mask, kernel, iterations=1)

        return pred_mask

    def save_segmentation_map(self, mask, key):
        save_path = os.path.join(self.out_folder, f"{key}_predict.png")
        cv2.imwrite(save_path, mask)

    def predict_segmentation_maps(self):
        predicted_dict = {}
        for key, img in self.images.items():
            pred_mask = self.predict_segmentation_map(img)
            predicted_dict[key] = pred_mask
            self.save_segmentation_map(pred_mask, key)
        return predicted_dict

#=========================<Your code ends here>==========================

if __name__ == "__main__":
    generator = RoadSegmentationMapGenerator(images_dict, output_folder)
    predicted_dict = generator.predict_segmentation_maps()

    avg_f1 = calculate_average_f1_score(predicted_dict, gt_dict)
    print(f"Average F1 score: {avg_f1:.4f}")
