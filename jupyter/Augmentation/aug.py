import os
import csv
import json
import random
import traceback
import multiprocessing
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =======================================================
# [1. 설정 및 상수]
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "IS_TEST_MODE": True,    # 테스트 모드 (True일 경우 TEST_COUNT만큼만 실행)
    "TEST_COUNT": 4,         # 테스트 실행 개수
    "NUM_WORKERS": 6,        # 병렬 프로세스 수
    
    # [경로 설정]
    "INPUT_ROOT": BASE_DIR,
    "OUTPUT_ROOT": os.path.join(BASE_DIR, "Augmentation_Output"),
    
    # [파일명 설정]
    "TARGET_CSV": "Seg_filtered.csv",
    "OUTPUT_CSV": "augmentation_log.csv",
}

# [증강 파이프라인 정의]
AUG_STEPS = [
    (1, 'pair', ['shear_L', 'shear_M']),
    (2, 'tgt_only', ['rot_L', 'rot_M', 'pers_L', 'pers_H']),
    (3, 'pair', ['hue_L', 'gray_L']),
    (4, 'tgt_only', ['elastic_L', 'elastic_M', 'elastic_H', 'bright_M', 'contrast_L', 'contrast_H', 'sat_L', 'eq_L', 'eq_H', 'noise_L', 'noise_M', 'stain_M'])
]

# 증강 파라미터 상세 설정
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
    current_pool = 1 
    total_generated = 0
    
    for _, _, methods in AUG_STEPS:
        num_methods = len(methods)
        generated_in_step = current_pool * num_methods
        total_generated += generated_in_step
        current_pool += generated_in_step
        
    total_expected = num_tasks * total_generated

    print("\n" + "="*50)
    print(f"📊 [통계 정보]")
    print("-" * 50)
    print(f" ▶ 대상 원본 쌍(Pair) : {num_tasks:,} 개")
    if CONFIG["IS_TEST_MODE"]:
        print(f"   (테스트 모드로 인해 최대 {CONFIG['TEST_COUNT']}개로 제한됨)")
    print(f" ▶ 1쌍당 생성 개수    : {total_generated:,} 개")
    print(f" ▶ 총 예상 생성 이미지 : {total_expected:,} 개")
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
# [5. 데이터 상태 관리 및 저장]
# =======================================================
class AugData:
    def __init__(self, ref_img, tgt_img, ref_name, tgt_name, meta, aug_method=None, aug_params=None):
        self.ref_img = ref_img
        self.tgt_img = tgt_img
        self.ref_name = ref_name
        self.tgt_name = tgt_name
        self.font = meta.get('font', '')
        self.logo = meta.get('logo', '')
        self.label_s = meta.get('label_s', 0)
        self.label_c = meta.get('label_c', 0)
        self.aug_method = aug_method if aug_method else ""
        self.aug_params = aug_params if aug_params else ""

    def update_label(self, method_tag):
        if self.label_s == 0 and (method_tag in FONT_NG_TRIGGERS):
            self.label_s = 1

    def _get_target_subfolder(self):
        s = self.label_s
        c = self.label_c
        if s == 1 and c == 1: return "font_diff_letter_diff"
        if s == 1 and c == 0: return "font_diff_letter_same"
        if s == 0 and c == 1: return "font_same_letter_diff"
        return "font_same_letter_same"

    def save_image(self, output_root):
        save_dir_ref = os.path.join(output_root, "ref_img")
        target_subfolder = self._get_target_subfolder()
        save_dir_tgt = os.path.join(output_root, "tar_img", target_subfolder)

        os.makedirs(save_dir_ref, exist_ok=True)
        os.makedirs(save_dir_tgt, exist_ok=True)

        ref_filename = f"{self.ref_name}.png"
        tgt_filename = f"{self.tgt_name}.png"
        abs_save_ref = os.path.join(save_dir_ref, ref_filename)
        abs_save_tgt = os.path.join(save_dir_tgt, tgt_filename)

        self.ref_img.save(abs_save_ref, compress_level=1)
        self.tgt_img.save(abs_save_tgt, compress_level=1)

        return abs_save_ref, abs_save_tgt

    def get_csv_row(self, saved_ref_path, saved_tgt_path):
        def to_relative(path):
            try:
                rel = os.path.relpath(path, CONFIG["OUTPUT_ROOT"])
                rel = rel.replace(os.sep, '/')
                if not rel.startswith('./'): rel = './' + rel
                return rel
            except: return path

        is_stained = 1 if self.aug_method == 'stain_M' else 0
        return {
            'tar_path': to_relative(saved_tgt_path),
            'ref_path': to_relative(saved_ref_path),
            'font': self.font,
            'logo': self.logo,
            'label_s': self.label_s,
            'label_c': self.label_c,
            'aug_method': self.aug_method,
            'aug_param': self.aug_params,
            'label_stain': is_stained
        }

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

        try:
            with Image.open(real_ref_path) as f: ref_img = f.convert("RGB"); ref_img.load()
            with Image.open(real_tgt_path) as f: tgt_img = f.convert("RGB"); tgt_img.load()
        except Exception:
            return []

        base = os.path.basename(ref_filename)
        base_ref_name = base[:-8] if base.lower().endswith('.png.jpg') else os.path.splitext(base)[0]
        
        stem = os.path.basename(tar_filename)
        stem = stem[:-8] if stem.lower().endswith('.png.jpg') else os.path.splitext(stem)[0]
        base_tgt_name = stem.split("@seg")[0] + "@seg" if "@seg" in stem else stem

        pool = [AugData(ref_img, tgt_img, base_ref_name, base_tgt_name, meta)]
        result_rows = []

        for _, scope, methods in AUG_STEPS:
            new_items = []
            for data in pool:
                for tag in methods:
                    next_meta = {
                        'font': data.font, 'logo': data.logo,
                        'label_s': data.label_s, 'label_c': data.label_c
                    }
                    
                    if scope == 'pair':
                        new_r, params_r = ImageAugmentor.apply_op(data.ref_img, tag)
                        new_t, params_t = ImageAugmentor.apply_op(data.tgt_img, tag)
                        name_r = f"{data.ref_name}@{tag}"
                        name_t = f"{data.tgt_name}@{tag}"
                        aug_info = json.dumps({"ref": params_r, "tgt": params_t}, ensure_ascii=False)
                    else:
                        new_r = data.ref_img.copy()
                        new_t, params_t = ImageAugmentor.apply_op(data.tgt_img, tag)
                        name_r = data.ref_name
                        name_t = f"{data.tgt_name}@{tag}"
                        aug_info = json.dumps({"ref": None, "tgt": params_t}, ensure_ascii=False)
                    
                    item = AugData(new_r, new_t, name_r, name_t, next_meta, aug_method=tag, aug_params=aug_info)
                    item.update_label(tag)
                    
                    s_ref, s_tgt = item.save_image(CONFIG["OUTPUT_ROOT"])
                    result_rows.append(item.get_csv_row(s_ref, s_tgt))
                    new_items.append(item)
            pool.extend(new_items)
            
        return result_rows

    except Exception:
        print(f"\n[Error processing {task_data.get('tar_filename')}]")
        traceback.print_exc()
        return []

# =======================================================
# [7. 메인 실행]
# =======================================================
def main():
    print(f"🚀 Augmentation Start (Workers: {CONFIG['NUM_WORKERS']})")
    
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

    print(f"✅ Finished. Saved to: {CONFIG['OUTPUT_ROOT']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    main()
