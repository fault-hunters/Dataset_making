import os
import csv
import json
import random
import traceback
import multiprocessing
import gc
import time
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps, ImageFile
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# [설정] 잘린 이미지 파일 로드 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =======================================================
# [1. 설정 및 상수]
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "IS_TEST_MODE": False,
    "TEST_COUNT": 4,
    "NUM_WORKERS": 30,  # CPU 코어 수에 맞춰 조절하세요
    
    "INPUT_ROOT": BASE_DIR,
    "OUTPUT_ROOT": os.path.join(BASE_DIR, "Augmentation_Output"),
    
    "TARGET_CSV": "Seg_filtered.csv",
    "OUTPUT_CSV": "augmentation_log.csv",
}

# [수정됨] 멘토 피드백 반영: shear_L을 tgt_only로 이동, Pair 단계에는 안전한 shear_M만 유지
AUG_STEPS = [
    (1, 'pair', ['shear_M']),
    (2, 'tgt_only', ['shear_L', 'rot_L', 'rot_M', 'pers_L', 'pers_H']),
    (3, 'pair', ['hue_L', 'gray_L']),
    (4, 'tgt_only', ['elastic_L', 'elastic_M', 'elastic_H', 'bright_M', 'contrast_L', 'contrast_H', 'sat_L', 'eq_L', 'eq_H', 'noise_L', 'noise_M', 'stain_M'])
]

PARAM_MAP = {
    "shear_L":   {"method": "shear", "range": (5, 15)},      
    "shear_M":   {"method": "shear", "range": (20, 30)},     
    "rot_L":     {"method": "rotate", "range": (5, 15)},     
    "rot_M":     {"method": "rotate", "range": (20, 30)},    
    "pers_L":    {"method": "perspective", "range": (0.01, 0.10), "fill": 255}, 
    "pers_H":    {"method": "perspective", "range": (0.22, 0.34), "fill": 255}, 
    "elastic_L": {"method": "elastic", "alpha": (15.0, 30.0),   "sigma": (4.0, 5.0)},
    "elastic_M": {"method": "elastic", "alpha": (50.0, 100.0),  "sigma": (5.0, 7.0)},
    "elastic_H": {"method": "elastic", "alpha": (120.0, 200.0), "sigma": (8.0, 10.0)},
    "hue_L":     {"method": "hue", "range": (0.01, 0.05)},
    "gray_L":    {"method": "grayscale", "range": (0.1, 0.3)},
    "bright_M":  {"method": "brightness", "range": (0.1, 0.2)},
    "contrast_L":{"method": "contrast", "range": (0.53, 0.80)},
    "contrast_H":{"method": "contrast", "range": (1.05, 1.32)},
    "sat_L":     {"method": "saturation", "range": (0.7, 0.95)},
    "eq_L":      {"method": "equalize", "range": (0.1, 0.3)},
    "eq_H":      {"method": "equalize", "range": (0.7, 0.9)},
    "noise_L":   {"method": "noise", "range": (0.01, 0.03)},
    "noise_M":   {"method": "noise", "range": (0.05, 0.1)},
    "stain_M":   {"method": "stain", "count": (10, 15), "opacity": (0.6, 0.8), "scale": (0.05, 0.10)}
}

# 라벨을 1(NG)로 변경하는 기법들 목록
FONT_NG_TRIGGERS = {"shear_L", "rot_L", "elastic_L", "elastic_M", "elastic_H"}

# =======================================================
# [2. 데이터 로더]
# =======================================================
class DataLoader:
    def __init__(self, config):
        self.config = config

    def get_absolute_path(self, relative_path_from_csv):
        if not relative_path_from_csv: return None
        clean_rel = relative_path_from_csv.strip().replace('\\', '/')
        if clean_rel.startswith('./'): clean_rel = clean_rel[2:]
        elif clean_rel.startswith('/'): clean_rel = clean_rel[1:]
        abs_path = os.path.join(self.config["INPUT_ROOT"], clean_rel)
        if os.path.exists(abs_path): return abs_path
        return None

    def create_tasks(self):
        csv_path = os.path.join(self.config["INPUT_ROOT"], self.config["TARGET_CSV"])
        if not os.path.exists(csv_path):
             csv_path = os.path.join(self.config["INPUT_ROOT"], "image_metadata", self.config["TARGET_CSV"])

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Target CSV not found: {self.config['TARGET_CSV']}")

        tasks = []
        print(f"📄 Reading target list from: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames: reader.fieldnames = [x.strip() for x in reader.fieldnames]

            for row in reader:
                real_ref = self.get_absolute_path(row.get('ref_path', ''))
                real_tar = self.get_absolute_path(row.get('tar_path', ''))

                if real_ref and real_tar:
                    task_meta = {
                        'font': row.get('font', ''), 
                        'logo': row.get('logo', ''),
                        'label_s': int(row.get('label_s', 0)),
                        'label_c': int(row.get('label_c', 0))
                    }
                    tasks.append({
                        'real_ref_path': real_ref,
                        'real_tar_path': real_tar,
                        'ref_filename': os.path.basename(real_ref),
                        'tar_filename': os.path.basename(real_tar),
                        'meta': task_meta
                    })
                
                if self.config["IS_TEST_MODE"] and len(tasks) >= self.config["TEST_COUNT"]:
                    break
        return tasks

# =======================================================
# [3. 통계 및 정보 출력]
# =======================================================
def print_statistics(task_list):
    num_tasks = len(task_list)
    # 아래 예상치는 참고용이며, 실제로는 라벨 상태에 따라 스킵되므로 실제 생성량은 이보다 적습니다.
    print("\n" + "="*50)
    print(f"📊 [작업 정보]")
    print("-" * 50)
    print(f" ▶ 대상 원본 쌍(Pair) : {num_tasks:,} 개")
    if CONFIG["IS_TEST_MODE"]:
        print(f"   (테스트 모드로 인해 최대 {CONFIG['TEST_COUNT']}개로 제한됨)")
    print(" ▶ 참고: 정상(OK) 이미지는 약 431배, 불량(NG) 이미지는 약 239배 증강됩니다.")
    print("=" * 50 + "\n")

# =======================================================
# [4. 이미지 증강 엔진]
# =======================================================
class ImageAugmentor:
    @staticmethod
    def apply_noise(img, severity_factor):
        img_tensor = v2.ToImage()(img)
        img_tensor = v2.ToDtype(torch.float32, scale=True)(img_tensor)
        noise = torch.randn_like(img_tensor) * severity_factor
        noisy_img = torch.clamp(img_tensor + noise, 0., 1.)
        return v2.ToPILImage()(noisy_img), {"severity": round(severity_factor, 4)}

    @staticmethod
    def add_clean_stain(pil_img, config):
        if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
        image = np.array(pil_img)
        h, w, _ = image.shape
        ink_layer = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_mask = np.zeros((h, w), dtype=np.float32)
        
        num_blobs = random.randint(*config["count"])
        global_opacity = random.uniform(*config["opacity"])
        
        for _ in range(num_blobs):
            cx, cy = random.randint(0, w), random.randint(0, h)
            min_s, max_s = config["scale"]
            axis_x = int(w * random.uniform(min_s, max_s))
            axis_y = int(h * random.uniform(min_s, max_s))
            color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            angle = random.randint(0, 360)
            cv2.ellipse(ink_layer, (cx, cy), (axis_x, axis_y), angle, 0, 360, color, -1)
            cv2.ellipse(alpha_mask, (cx, cy), (axis_x, axis_y), angle, 0, 360, 1.0, -1)
            
        k_size = (21, 21)
        ink_blurred = cv2.GaussianBlur(ink_layer, k_size, 0)
        mask_blurred = cv2.GaussianBlur(alpha_mask, k_size, 0)
        mask_blurred[mask_blurred < 0.05] = 0.0
        
        final_alpha = mask_blurred * global_opacity
        final_alpha_3ch = cv2.merge([final_alpha, final_alpha, final_alpha])
        image_float = image.astype(np.float32)
        ink_float = ink_blurred.astype(np.float32)
        output = image_float * (1.0 - final_alpha_3ch) + ink_float * final_alpha_3ch
        
        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), {"count": num_blobs, "opacity": round(global_opacity, 3)}

    @classmethod
    def apply_op(cls, img, tag):
        if tag not in PARAM_MAP: return img, {}
        config = PARAM_MAP[tag]
        method = config["method"]
        processed = img.copy()
        params_log = {}

        if method == "shear":
            min_v, max_v = config["range"]
            val = random.uniform(min_v, max_v) * random.choice([-1, 1])
            axis = random.choice(['x', 'y'])
            processed = F.affine(processed, angle=0, translate=[0,0], scale=1.0, 
                                 shear=[val, 0.0] if axis=='x' else [0.0, val], 
                                 interpolation=v2.InterpolationMode.BILINEAR, fill=255)
            params_log = {"axis": axis, "val": round(val, 2)}
        elif method == "rotate":
            min_v, max_v = config["range"]
            val = random.uniform(min_v, max_v) * random.choice([-1, 1])
            processed = F.rotate(processed, angle=val, interpolation=v2.InterpolationMode.BILINEAR, fill=255)
            params_log = {"angle": round(val, 2)}
        elif method == "perspective":
            min_v, max_v = config["range"]
            scale = random.uniform(min_v, max_v)
            processed = v2.RandomPerspective(distortion_scale=scale, p=1.0, fill=config.get("fill", 255))(processed)
            params_log = {"distortion_scale": round(scale, 3)}
        elif method == "elastic":
            alpha = random.uniform(*config["alpha"])
            sigma = random.uniform(*config["sigma"])
            processed = v2.ElasticTransform(alpha=alpha, sigma=sigma)(processed)
            params_log = {"alpha": round(alpha, 1), "sigma": round(sigma, 1)}
        elif method == "hue":
            min_v, max_v = config["range"]
            val = max(-0.5, min(random.uniform(min_v, max_v) * random.choice([-1, 1]), 0.5))
            processed = F.adjust_hue(processed, val)
            params_log = {"hue_factor": round(val, 3)}
        elif method == "grayscale":
            alpha = random.uniform(*config["range"])
            if processed.mode != 'RGB': processed = processed.convert('RGB')
            processed = Image.blend(processed, ImageOps.grayscale(processed).convert("RGB"), alpha)
            params_log = {"gray_alpha": round(alpha, 2)}
        elif method == "brightness":
            min_v, max_v = config["range"]
            factor = max(0.0, 1.0 + (random.uniform(min_v, max_v) * random.choice([-1, 1])))
            processed = F.adjust_brightness(processed, factor)
            params_log = {"bright_factor": round(factor, 2)}
        elif method == "contrast":
            val = random.uniform(*config["range"])
            processed = F.adjust_contrast(processed, val)
            params_log = {"contrast_factor": round(val, 2)}
        elif method == "saturation":
            val = random.uniform(*config["range"])
            processed = F.adjust_saturation(processed, val)
            params_log = {"sat_factor": round(val, 2)}
        elif method == "equalize":
            alpha = random.uniform(*config["range"])
            if processed.mode != 'RGB': processed = processed.convert('RGB')
            processed = Image.blend(processed, ImageOps.equalize(processed), alpha)
            params_log = {"eq_alpha": round(alpha, 2)}
        elif method == "noise":
            val = random.uniform(*config["range"])
            processed, p_log = cls.apply_noise(processed, val)
            params_log = p_log
        elif method == "stain":
            processed, p_log = cls.add_clean_stain(processed, config)
            params_log = p_log

        return processed, params_log

# =======================================================
# [5. 데이터 상태 관리]
# =======================================================
class AugData:
    def __init__(self, ref_path, tgt_path, ref_name, tgt_name, meta, aug_method=None, aug_params=None):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.ref_name = ref_name
        self.tgt_name = tgt_name
        self.font = meta.get('font', '')
        self.logo = meta.get('logo', '')
        self.label_s = meta.get('label_s', 0)
        self.label_c = meta.get('label_c', 0)
        self.aug_method = aug_method if aug_method else ""
        self.aug_params = aug_params if aug_params else ""

    def update_label(self, method_tag):
        # 만약 NG를 유발하는 기법(Trigger)이 적용되었다면, 라벨을 1로 변경
        if self.label_s == 0 and (method_tag in FONT_NG_TRIGGERS):
            self.label_s = 1

    def get_target_subfolder(self):
        s = self.label_s
        c = self.label_c
        if s == 1 and c == 1: return "font_diff_letter_diff"
        if s == 1 and c == 0: return "font_diff_letter_same"
        if s == 0 and c == 1: return "font_same_letter_diff"
        return "font_same_letter_same"

    def get_csv_row(self):
        def to_relative(path):
            try:
                if path is None: return ""
                rel = os.path.relpath(path, CONFIG["OUTPUT_ROOT"])
                rel = rel.replace(os.sep, '/')
                if not rel.startswith('./'): rel = './' + rel
                return rel
            except: return path

        is_stained = 1 if self.aug_method == 'stain_M' else 0
        return {
            'tar_path': to_relative(self.tgt_path),
            'ref_path': to_relative(self.ref_path),
            'font': self.font,
            'logo': self.logo,
            'label_s': self.label_s,
            'label_c': self.label_c,
            'aug_method': self.aug_method,
            'aug_param': self.aug_params,
            'label_stain': is_stained
        }

# [헬퍼 함수]
def save_image_immediate(img, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    img.save(save_path, compress_level=1)
    return save_path

def load_image_with_retry(path, retries=5, delay=0.2):
    for i in range(retries):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
                
            with Image.open(path) as f:
                img = f.convert("RGB")
                img.load()
                return img
        except (OSError, IOError, FileNotFoundError) as e:
            if i == retries - 1:
                raise e 
            time.sleep(delay + random.uniform(0, 0.1))
    return None

# =======================================================
# [6. 병렬 Worker 함수]
# =======================================================
def worker_process(task_data):
    try:
        real_ref_path = task_data['real_ref_path']
        real_tgt_path = task_data['real_tar_path']
        ref_filename = task_data['ref_filename']
        tar_filename = task_data['tar_filename']
        meta = task_data['meta']

        base = os.path.basename(ref_filename)
        base_ref_name = base[:-8] if base.lower().endswith('.png.jpg') else os.path.splitext(base)[0]
        
        stem = os.path.basename(tar_filename)
        stem = stem[:-8] if stem.lower().endswith('.png.jpg') else os.path.splitext(stem)[0]
        base_tgt_name = stem.split("@seg")[0] + "@seg" if "@seg" in stem else stem

        pool = [AugData(real_ref_path, real_tgt_path, base_ref_name, base_tgt_name, meta)]
        result_rows = []

        for step_idx, scope, methods in AUG_STEPS:
            new_items = []
            
            for data in pool:
                try:
                    src_ref_img = load_image_with_retry(data.ref_path)
                    src_tgt_img = load_image_with_retry(data.tgt_path)
                    
                    if src_ref_img is None or src_tgt_img is None:
                        raise IOError("Failed to load image after retries")
                        
                except Exception as e:
                    print(f"\n[Read Error] Path: {data.tgt_path}\nReason: {e}")
                    continue

                for tag in methods:
                    # =======================================================
                    # [핵심 수정] 이중 NG 방지 로직
                    # 이미 라벨이 1(NG)인데, 또 라벨을 1로 만드는 기법이면 건너뜀
                    # =======================================================
                    if (data.label_s == 1) and (tag in FONT_NG_TRIGGERS):
                        continue

                    next_meta = {
                        'font': data.font, 'logo': data.logo,
                        'label_s': data.label_s, 'label_c': data.label_c
                    }
                    
                    if scope == 'pair':
                        res_ref_img, params_r = ImageAugmentor.apply_op(src_ref_img, tag)
                        res_tgt_img, params_t = ImageAugmentor.apply_op(src_tgt_img, tag)
                        name_r = f"{data.ref_name}@{tag}"
                        name_t = f"{data.tgt_name}@{tag}"
                        aug_info = json.dumps({"ref": params_r, "tgt": params_t}, ensure_ascii=False)
                    else:
                        res_ref_img = src_ref_img.copy()
                        res_tgt_img, params_t = ImageAugmentor.apply_op(src_tgt_img, tag)
                        name_r = data.ref_name
                        name_t = f"{data.tgt_name}@{tag}"
                        aug_info = json.dumps({"ref": None, "tgt": params_t}, ensure_ascii=False)
                    
                    item = AugData(None, None, name_r, name_t, next_meta, aug_method=tag, aug_params=aug_info)
                    
                    # 라벨 업데이트 (0 -> 1 변경 수행)
                    item.update_label(tag)
                    
                    target_subfolder = item.get_target_subfolder()
                    save_dir_ref = os.path.join(CONFIG["OUTPUT_ROOT"], "ref_img")
                    save_dir_tgt = os.path.join(CONFIG["OUTPUT_ROOT"], "tar_img", target_subfolder)
                    
                    filename_ref = f"{name_r}.png"
                    filename_tgt = f"{name_t}.png"

                    saved_ref_path = save_image_immediate(res_ref_img, save_dir_ref, filename_ref)
                    saved_tgt_path = save_image_immediate(res_tgt_img, save_dir_tgt, filename_tgt)
                    
                    item.ref_path = saved_ref_path
                    item.tgt_path = saved_tgt_path
                    
                    result_rows.append(item.get_csv_row())
                    new_items.append(item)

                    del res_ref_img
                    del res_tgt_img
                
                del src_ref_img
                del src_tgt_img

            pool.extend(new_items)
            gc.collect()
            
        return result_rows

    except Exception:
        print(f"\n[Critical Error processing {task_data.get('tar_filename')}]")
        traceback.print_exc()
        return []

# =======================================================
# [7. 메인 실행]
# =======================================================
def main():
    multiprocessing.freeze_support()
    
    print(f"🚀 Augmentation Start (Workers: {CONFIG['NUM_WORKERS']})")
    print(f"💾 Mode: Anti-Double-NG + Safe Pair Augmentation")
    
    loader = DataLoader(CONFIG)
    
    try:
        task_list = loader.create_tasks()
    except Exception as e:
        print(f"❌ Error creating tasks: {e}")
        return

    if not task_list:
        print("❌ No valid tasks found.")
        return

    print_statistics(task_list)

    os.makedirs(CONFIG["OUTPUT_ROOT"], exist_ok=True)
    output_csv = os.path.join(CONFIG["OUTPUT_ROOT"], CONFIG["OUTPUT_CSV"])
    
    headers = ['tar_path', 'ref_path', 'font', 'logo', 'label_s', 'label_c', 
               'aug_method', 'aug_param', 'label_stain']

    print("⏳ Processing...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=headers)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
            futures = [executor.submit(worker_process, task) for task in task_list]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting"):
                results = future.result()
                if results:
                    writer.writerows(results)
                    out_f.flush()

    print(f"✅ Finished. Saved to: {CONFIG['OUTPUT_ROOT']}")

if __name__ == "__main__":
    main()