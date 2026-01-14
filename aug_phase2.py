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

# [안전장치] 잘린 이미지 로드 방지
ImageFile.LOAD_TRUNCATED_IMAGES = False

# =======================================================
# [1. 설정 및 상수]
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "IS_TEST_MODE": False,
    "TEST_COUNT": 5,  # 테스트 시 5장만 수행
    "NUM_WORKERS": 30,  # CPU/RAM 사양에 따라 조절 (안정성 권장: 10~15)
    "MAX_RAM_CACHE": 30,

    "INPUT_ROOT": BASE_DIR,
    "OUTPUT_ROOT": os.path.join(BASE_DIR, "final_output"),

    "TARGET_CSV": "Seg_filtered.csv",
    "OUTPUT_CSV": "augmentation_log.csv",
    "CHECKPOINT_FILE": "completed_tasks.txt"
}

# [수정됨] 요청하신 OK/NG 로직 및 제외 항목 반영
AUG_STEPS = [
    # Step 1: Pair 적용 (OK 기하학)
    (1, 'pair', ['shear_M', 'rot_M']),

    # Step 2: Target Only 적용 (NG 기하학 - 미세한 비틀림이 NG로 간주될 경우)
    (2, 'tgt_only', ['shear_L', 'rot_L']),

    # Step 3: Pair 적용 (OK 색상)
    (3, 'pair', ['hue_L', 'gray_L']),

    # Step 4: Target Only 적용 (NG 왜곡 + OK 화질저하)
    (4, 'tgt_only', ['elastic_L', 'elastic_H', 'bright_M', 'contrast_L', 'eq_H', 'noise_L'])
]

# [수정됨] 사용하지 않는 파라미터 제거 및 정리
PARAM_MAP = {
    # --- OK Group ---
    "shear_M": {"method": "shear", "range": (20, 30)},
    "rot_M": {"method": "rotate", "range": (20, 30)},
    "hue_L": {"method": "hue", "range": (0.01, 0.05)},
    "gray_L": {"method": "grayscale", "range": (0.1, 0.3)},
    "bright_M": {"method": "brightness", "range": (0.1, 0.2)},  # Low~Mid 커버
    "contrast_L": {"method": "contrast", "range": (0.53, 0.80)},
    "eq_H": {"method": "equalize", "range": (0.7, 0.9)},
    "noise_L": {"method": "noise", "range": (0.01, 0.03)},

    # --- NG Group ---
    "shear_L": {"method": "shear", "range": (5, 15)},
    "rot_L": {"method": "rotate", "range": (5, 15)},
    "elastic_L": {"method": "elastic", "alpha": (15.0, 30.0), "sigma": (4.0, 5.0)},
    "elastic_H": {"method": "elastic", "alpha": (120.0, 200.0), "sigma": (8.0, 10.0)}
}

# [수정됨] NG 라벨(1)을 유발하는 트리거 리스트
FONT_NG_TRIGGERS = {"shear_L", "rot_L", "elastic_L", "elastic_H"}


# =======================================================
# [2. 데이터 로더]
# =======================================================
class DataLoader:
    def __init__(self, config):
        self.config = config

    def get_absolute_path(self, relative_path_from_csv):
        if not relative_path_from_csv: return None
        clean_rel = relative_path_from_csv.strip().replace('\\', '/')
        if clean_rel.startswith('./'):
            clean_rel = clean_rel[2:]
        elif clean_rel.startswith('/'):
            clean_rel = clean_rel[1:]
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
# [3. 헬퍼 함수]
# =======================================================
def save_image_immediate(img, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    img.save(save_path, compress_level=1)
    return save_path

def is_black_image(pil_img, threshold=5.0):
    """
    이미지가 검은색(또는 거의 검은색)인지 판별합니다.
    threshold: 평균 픽셀값 기준 (0~255 중 5 미만이면 검은색으로 간주)
    """
    if pil_img is None: return True
    try:
        # 이미지를 numpy 배열로 변환
        img_arr = np.array(pil_img)
        # 평균 픽셀 값이 threshold보다 낮으면 검은색 이미지로 판단
        return np.mean(img_arr) < threshold
    except Exception:
        return True

def load_image_with_retry(path, retries=3, delay=0.2):
    for i in range(retries):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            with Image.open(path) as f:
                img = f.convert("RGB")
                img.load()
                return img
        except Exception as e:
            if i == retries - 1:
                return None
            time.sleep(delay)
    return None


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

    # [수정됨] Stain, Perspective 관련 함수 및 분기 제거
    @classmethod
    def apply_op(cls, img, tag, manual_param=None):
        if tag not in PARAM_MAP: return img, {}
        config = PARAM_MAP[tag]
        method = config["method"]
        processed = img.copy()
        params_log = {}

        if method == "shear":
            min_v, max_v = config["range"]
            if manual_param is not None:
                val, axis = manual_param
            else:
                val = random.uniform(min_v, max_v) * random.choice([-1, 1])
                axis = random.choice(['x', 'y'])
            processed = F.affine(processed, angle=0, translate=[0, 0], scale=1.0,
                                 shear=[val, 0.0] if axis == 'x' else [0.0, val],
                                 interpolation=v2.InterpolationMode.BILINEAR, fill=255)
            params_log = {"axis": axis, "val": round(val, 2)}

        elif method == "rotate":
            min_v, max_v = config["range"]
            if manual_param is not None:
                val = manual_param
            else:
                val = random.uniform(min_v, max_v) * random.choice([-1, 1])
            processed = F.rotate(processed, angle=val, interpolation=v2.InterpolationMode.BILINEAR, fill=255)
            params_log = {"angle": round(val, 2)}

        elif method == "elastic":
            alpha = random.uniform(*config["alpha"])
            sigma = random.uniform(*config["sigma"])
            processed = v2.ElasticTransform(alpha=alpha, sigma=sigma)(processed)
            params_log = {"alpha": round(alpha, 1), "sigma": round(sigma, 1)}

        elif method == "hue":
            min_v, max_v = config["range"]
            if manual_param is not None:
                val = manual_param
            else:
                val = max(-0.5, min(random.uniform(min_v, max_v) * random.choice([-1, 1]), 0.5))
            processed = F.adjust_hue(processed, val)
            params_log = {"hue_factor": round(val, 3)}

        elif method == "grayscale":
            if manual_param is not None:
                alpha = manual_param
            else:
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

        elif method == "equalize":
            alpha = random.uniform(*config["range"])
            if processed.mode != 'RGB': processed = processed.convert('RGB')
            processed = Image.blend(processed, ImageOps.equalize(processed), alpha)
            params_log = {"eq_alpha": round(alpha, 2)}

        elif method == "noise":
            val = random.uniform(*config["range"])
            processed, p_log = cls.apply_noise(processed, val)
            params_log = p_log

        return processed, params_log

    @staticmethod
    def generate_seed_param(tag):
        if tag not in PARAM_MAP: return None
        config = PARAM_MAP[tag]
        method = config["method"]
        if method == "shear":
            min_v, max_v = config["range"]
            return (random.uniform(min_v, max_v) * random.choice([-1, 1]), random.choice(['x', 'y']))
        elif method == "rotate":
            min_v, max_v = config["range"]
            return random.uniform(min_v, max_v) * random.choice([-1, 1])
        elif method == "hue":
            min_v, max_v = config["range"]
            return max(-0.5, min(random.uniform(min_v, max_v) * random.choice([-1, 1]), 0.5))
        elif method == "grayscale":
            return random.uniform(*config["range"])
        return None


# =======================================================
# [5. 데이터 객체]
# =======================================================
class AugData:
    def __init__(self, ref_path, tgt_path, ref_name, tgt_name, meta,
                 aug_method=None, aug_params=None,
                 img_ref_obj=None, img_tgt_obj=None):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.ref_name = ref_name
        self.tgt_name = tgt_name
        self.meta = meta
        self.aug_method = aug_method if aug_method else ""
        self.aug_params = aug_params if aug_params else ""
        self.img_ref_obj = img_ref_obj
        self.img_tgt_obj = img_tgt_obj

    def get_images(self):
        if self.img_ref_obj is None:
            self.img_ref_obj = load_image_with_retry(self.ref_path)
        if self.img_tgt_obj is None:
            self.img_tgt_obj = load_image_with_retry(self.tgt_path)
        return self.img_ref_obj, self.img_tgt_obj

    def release_memory(self):
        self.img_ref_obj = None
        self.img_tgt_obj = None

    def update_label(self, method_tag):
        if self.meta['label_s'] == 0 and (method_tag in FONT_NG_TRIGGERS):
            self.meta['label_s'] = 1

    def get_target_subfolder(self):
        s = self.meta['label_s']
        c = self.meta['label_c']
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
            except:
                return path

        return {
            'tar_path': to_relative(self.tgt_path),
            'ref_path': to_relative(self.ref_path),
            'font': self.meta.get('font', ''),
            'logo': self.meta.get('logo', ''),
            'label_s': self.meta.get('label_s', 0),
            'label_c': self.meta.get('label_c', 0),
            'aug_method': self.aug_method,
            'aug_param': self.aug_params,
            'label_stain': 0  # Stain 기능 제거로 항상 0
        }


# =======================================================
# [6. 워커 프로세스]
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

        origin_ref_img = load_image_with_retry(real_ref_path)
        origin_tgt_img = load_image_with_retry(real_tgt_path)

        if origin_ref_img is None or origin_tgt_img is None:
            return []

        pool = [AugData(real_ref_path, real_tgt_path, base_ref_name, base_tgt_name, meta,
                        img_ref_obj=origin_ref_img, img_tgt_obj=origin_tgt_img)]

        result_rows = []

        for step_idx, scope, methods in AUG_STEPS:
            new_items = []

            if len(pool) > CONFIG["MAX_RAM_CACHE"]:
                for old_item in pool[:-CONFIG["MAX_RAM_CACHE"]]:
                    old_item.release_memory()
                gc.collect()

            for data in pool:
                src_ref_img, src_tgt_img = data.get_images()
                if src_ref_img is None or src_tgt_img is None: continue

                for tag in methods:
                    # [수정] 재시도 로직 추가 (최대 3회)
                    retry_count = 0
                    max_retries = 3

                    res_ref_img = None
                    res_tgt_img = None
                    params_r = {}
                    params_t = {}
                    aug_info = ""
                    success = False

                    while retry_count < max_retries:
                        # 1. 파라미터 생성 및 적용
                        if scope == 'pair':
                            seed_param = ImageAugmentor.generate_seed_param(tag)
                            res_ref_img, params_r = ImageAugmentor.apply_op(src_ref_img, tag, manual_param=seed_param)
                            res_tgt_img, params_t = ImageAugmentor.apply_op(src_tgt_img, tag, manual_param=seed_param)
                            name_r = f"{data.ref_name}@{tag}"
                            name_t = f"{data.tgt_name}@{tag}"
                            aug_info = json.dumps({"ref": params_r, "tgt": params_t}, ensure_ascii=False)
                        else:
                            res_ref_img = src_ref_img
                            res_tgt_img, params_t = ImageAugmentor.apply_op(src_tgt_img, tag)
                            name_r = data.ref_name
                            name_t = f"{data.tgt_name}@{tag}"
                            aug_info = json.dumps({"ref": None, "tgt": params_t}, ensure_ascii=False)

                        # 2. [검증] 검은 이미지 체크
                        if is_black_image(res_ref_img) or is_black_image(res_tgt_img):
                            retry_count += 1
                            # print(f"⚠️ Black image detected ({tag}). Retrying {retry_count}/{max_retries}...")
                            continue  # 다시 while문 처음으로 돌아가서 새로운 파라미터로 시도
                        else:
                            success = True
                            break  # 정상 이미지면 while 탈출

                    # 3회 시도 후에도 검은색이면 건너뜀 (데이터 오염 방지)
                    if not success:
                        # print(f"❌ Skipping {tag} due to persistent black image error.")
                        continue

                    # --- 이하 저장 로직은 기존과 동일 ---
                    next_meta = data.meta.copy()

                    # 라벨 업데이트 (NG 조건 발생 시 label_s = 1)
                    if next_meta['label_s'] == 0 and (tag in FONT_NG_TRIGGERS):
                        next_meta['label_s'] = 1

                    temp_item = AugData(None, None, name_r, name_t, next_meta, aug_method=tag, aug_params=aug_info)
                    target_subfolder = temp_item.get_target_subfolder()

                    # (이하 저장 코드 동일...)
                    save_dir_ref = os.path.join(CONFIG["OUTPUT_ROOT"], "ref_img")
                    save_dir_tgt = os.path.join(CONFIG["OUTPUT_ROOT"], "tar_img", target_subfolder)

                    filename_ref = f"{name_r}.png"
                    filename_tgt = f"{name_t}.png"

                    saved_ref_path = save_image_immediate(res_ref_img, save_dir_ref, filename_ref)
                    saved_tgt_path = save_image_immediate(res_tgt_img, save_dir_tgt, filename_tgt)

                    next_item = AugData(saved_ref_path, saved_tgt_path, name_r, name_t, next_meta,
                                        aug_method=tag, aug_params=aug_info,
                                        img_ref_obj=res_ref_img, img_tgt_obj=res_tgt_img)

                    result_rows.append(next_item.get_csv_row())
                    new_items.append(next_item)

            pool.extend(new_items)

        del pool
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

    os.makedirs(CONFIG["OUTPUT_ROOT"], exist_ok=True)
    checkpoint_path = os.path.join(CONFIG["OUTPUT_ROOT"], CONFIG["CHECKPOINT_FILE"])
    output_csv = os.path.join(CONFIG["OUTPUT_ROOT"], CONFIG["OUTPUT_CSV"])

    completed_tasks = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            completed_tasks = set(line.strip() for line in f)
        print(f"🔄 Resuming... Found {len(completed_tasks)} completed tasks.")
    else:
        print(f"🆕 Starting fresh.")

    print(f"🚀 Augmentation Start (Workers: {CONFIG['NUM_WORKERS']})")

    loader = DataLoader(CONFIG)
    try:
        all_tasks = loader.create_tasks()
    except Exception as e:
        print(f"❌ Error creating tasks: {e}")
        return

    task_list = [t for t in all_tasks if t['tar_filename'] not in completed_tasks]

    if not task_list:
        print("✅ All tasks are already completed!")
        return

    print(f"📋 Remaining tasks: {len(task_list)} / {len(all_tasks)}")

    headers = ['tar_path', 'ref_path', 'font', 'logo', 'label_s', 'label_c',
               'aug_method', 'aug_param', 'label_stain']

    file_mode = 'a' if (completed_tasks and os.path.exists(output_csv)) else 'w'

    print("⏳ Processing...")

    with open(output_csv, file_mode, newline='', encoding='utf-8-sig') as out_f, \
            open(checkpoint_path, 'a', encoding='utf-8') as cp_f:

        writer = csv.DictWriter(out_f, fieldnames=headers)
        if file_mode == 'w':
            writer.writeheader()

        with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
            future_to_id = {
                executor.submit(worker_process, task): task['tar_filename']
                for task in task_list
            }

            for future in tqdm(as_completed(future_to_id), total=len(task_list), desc="Augmenting"):
                task_id = future_to_id[future]
                try:
                    results = future.result()
                    if results:
                        writer.writerows(results)
                        out_f.flush()
                        cp_f.write(task_id + '\n')
                        cp_f.flush()
                except Exception as e:
                    print(f"❌ Error in task {task_id}: {e}")

    print(f"✅ Finished. Saved to: {CONFIG['OUTPUT_ROOT']}")


if __name__ == "__main__":
    main()