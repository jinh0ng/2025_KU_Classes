import os
import cv2
import numpy as np
from sklearn.metrics import f1_score
from abc import ABCMeta, abstractmethod
import requests
import zipfile

# 폴더 경로 설정
input_folder = "./exercise_input"
output_folder = "./exercise_output"

# 데이터셋 다운로드 및 압축 해제 함수


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

    if download_success:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(input_folder)
            print(f"Decompression complete!: {input_folder}")
            os.remove(zip_path)
            print(f"ZIP file deletion completed: {zip_path}")
        except zipfile.BadZipFile:
            print("Decompression failed: The ZIP file may be corrupted.")

# 입력 이미지 로드 함수 (정답 마스크 파일 제외)


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

# 정답(ground truth) 마스크 로드 함수 ("_map" 또는 "_road" 포함된 파일)


def load_ground_truth_dict(input_folder):
    gt_dict = {}
    for filename in os.listdir(input_folder):
        if "_map" in filename:
            name, ext = os.path.splitext(filename)
            key_name = name.replace("_map", "").replace("_road", "")
            filepath = os.path.join(input_folder, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # 127 이상은 255, 미만은 0으로 이진화
                _, binary_mask = cv2.threshold(
                    mask, 127, 255, cv2.THRESH_BINARY)
                gt_dict[key_name] = binary_mask
    return gt_dict

# 예측 마스크와 정답 마스크를 비교하여 평균 F1 스코어를 계산하는 함수


def calculate_average_f1_score(predicted_dict, gt_dict):
    f1_scores = []
    common_keys = set(predicted_dict.keys()).intersection(set(gt_dict.keys()))

    for key in common_keys:
        pred_mask = predicted_dict[key]
        gt_mask = gt_dict[key]

        # 예측 마스크와 정답 마스크 크기가 다르면 정답 마스크 크기에 맞게 조정
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 127은 'don’t care' 영역으로 간주하여 평가에서 제외
        valid_mask = (gt_mask != 127)
        pred_flat = (pred_mask.flatten() > 127).astype(np.uint8)
        gt_flat = (gt_mask.flatten() > 127).astype(np.uint8)
        valid_flat = valid_mask.flatten()

        if np.any(valid_flat):
            score = f1_score(gt_flat[valid_flat], pred_flat[valid_flat])
            f1_scores.append(score)

    if len(f1_scores) == 0:
        return 0.0
    return np.mean(f1_scores)


# 데이터셋이 없으면 다운로드 및 압축 해제
if not os.path.exists(input_folder):
    download_dataset()

# 출력 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 입력 이미지와 정답 마스크 로드
images_dict = load_images_dict(input_folder)
print(f"Test set: {len(images_dict)} images")

gt_dict = load_ground_truth_dict(input_folder)
print(f"Segmentation mask: {len(gt_dict)} labels")


class SegmentationMapGenerator(metaclass=ABCMeta):
    @abstractmethod
    def predict_segmentation_map(self, img):
        pass

    @abstractmethod
    def predict_segmentation_maps(self):
        pass

# =========================<Your code starts here>========================


class RoadSegmentationMapGenerator(SegmentationMapGenerator):
    def __init__(self, images, out_folder):
        self.images = images
        self.out_folder = out_folder

    def predict_segmentation_map(self, img):
        # --------- Preprocessing: Bilateral Filtering ---------
        filtered = cv2.bilateralFilter(img, d=6, sigmaColor=100, sigmaSpace=80)

        # --------- Pipeline 1: LAB 기반 분할 ---------
        # LAB 변환 및 회색(도로) 영역 추출
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        lower_lab = np.array([45, 120, 120])
        upper_lab = np.array([200, 135, 135])

        lab_mask = cv2.inRange(lab, lower_lab, upper_lab)
        # L 채널 평활화와 Otsu 이진화
        L_channel, _, _ = cv2.split(lab)
        eq_L = cv2.equalizeHist(L_channel)
        _, otsu_mask = cv2.threshold(
            eq_L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 두 마스크의 공통 영역
        combined_lab = cv2.bitwise_and(lab_mask, otsu_mask)
        # Morphological 연산: closing → opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        morph_lab = cv2.morphologyEx(
            combined_lab, cv2.MORPH_CLOSE, kernel, iterations=3)
        morph_lab = cv2.morphologyEx(
            morph_lab, cv2.MORPH_OPEN, kernel, iterations=3)
        # 윤곽선 분석: 가장 큰 연결 영역 선택
        contours, _ = cv2.findContours(
            morph_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_lab_final = np.zeros_like(morph_lab)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_lab_final, [
                             max_contour], -1, 255, thickness=cv2.FILLED)
        else:
            mask_lab_final = morph_lab
        # 경계 부드럽게: dilation
        mask_lab_final = cv2.dilate(mask_lab_final, kernel, iterations=3)

        # --------- Pipeline 2: HSV 기반 분할 ---------
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        # 채도는 낮고 명도는 중간인 영역 (도로의 회색 특성)
        mask_s = cv2.inRange(s_channel, 0, 55)
        mask_v = cv2.inRange(v_channel, 75, 245)
        mask_hsv = cv2.bitwise_and(mask_s, mask_v)
        # ROI 제한: 도로는 주로 하단에 위치하므로 이미지 하단부분분만 사용
        h_img, w_img = mask_hsv.shape
        roi_mask = np.zeros_like(mask_hsv)
        roi_mask[int(h_img * 0.635):, :] = 255
        mask_hsv = cv2.bitwise_and(mask_hsv, roi_mask)
        # Morphological 연산: Closing과 Opening
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask_hsv = cv2.morphologyEx(
            mask_hsv, cv2.MORPH_CLOSE, kernel_rect, iterations=1)
        mask_hsv = cv2.morphologyEx(
            mask_hsv, cv2.MORPH_OPEN, kernel_rect, iterations=1)
        # 연결 영역 분석: 가장 큰 영역 선택
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_hsv, connectivity=8)
        mask_hsv_final = np.zeros_like(mask_hsv)
        if num_labels > 1:
            max_area = 0
            max_label = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_label = i
            mask_hsv_final = (labels == max_label).astype(np.uint8) * 255
        else:
            mask_hsv_final = mask_hsv
        mask_hsv_final = cv2.dilate(mask_hsv_final, kernel_rect, iterations=1)

        # --------- 앙상블: 두 파이프라인 결과 결합 ---------
        combined_mask = cv2.bitwise_or(mask_lab_final, mask_hsv_final)

        # --------- Final Post-processing ---------
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_OPEN, kernel_final, iterations=7)
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel_final, iterations=7)

        # 연결 컴포넌트 분석으로 작은 잡음 제거
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8)
        final_mask = np.zeros_like(combined_mask)
        min_area = 1000  # 데이터셋 특성에 따라 조절
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                final_mask[labels == i] = 255

        return final_mask

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


# =========================<Your code ends here>==========================


if __name__ == "__main__":
    generator = RoadSegmentationMapGenerator(images_dict, output_folder)
    predicted_dict = generator.predict_segmentation_maps()

    avg_f1 = calculate_average_f1_score(predicted_dict, gt_dict)
    print(f"Average F1 score: {avg_f1:.4f}")
