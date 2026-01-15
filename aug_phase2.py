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
    "IS_TEST_MODE": False,  # 전체 실행 시 False
    "TEST_COUNT": 5,
    "NUM_WORKERS": 10,
    "MAX_RAM_CACHE": 5,
    "INPUT_ROOT": BASE_DIR,
    "OUTPUT_ROOT": os.path.join(BASE_DIR, "phase2_output_csv"),

    # [수정 1] 대상 CSV 파일명 변경
    "TARGET_CSV": "phase2_pairs_masked_.csv",
    "OUTPUT_CSV": "augmentation_log.csv",
    "CHECKPOINT_FILE": "completed_tasks.txt",

    # [수정 2] CSV 헤더 매핑 수정 (before 컬럼은 제외됨)
    "CSV_COL_MAP": {
        "REF_IMG": "ref_image",  # CSV 헤더: ref_image
        "TAR_IMG": "tar_image",  # CSV 헤더: tar_image
        "REF_MASK": "ref_masked",  # CSV 헤더: ref_masked
        "TAR_MASK": "tar_masked",  # CSV 헤더: tar_masked
    }
}

# [증강 파이프라인] 4단계 (총 81배 증강, 완벽 동기화)
AUG_STEPS = [
    (1, 'pair', ['shear_M', 'rot_M']),
    (2, 'tgt_only', ['bright_M', 'contrast_L']),
    (3, 'pair', ['hue_L', 'gray_L']),
    (4, 'tgt_only', ['noise_L', 'eq_H'])
]

# [파라미터 설정]
PARAM_MAP = {
    "shear_M": {"method": "shear", "range": (20, 30)},
    "rot_M": {"method": "rotate", "range": (20, 30)},
    "hue_L": {"method": "hue", "range": (0.01, 0.05)},
    "gray_L": {"method": "grayscale", "range": (0.1, 0.3)},
    "bright_M": {"method": "brightness", "range": (0.1, 0.2)},
    "contrast_L": {"method": "contrast", "range": (0.53, 0.80)},
    "eq_H": {"method": "equalize", "range": (0.7, 0.9)},
    "noise_L": {"method": "noise", "range": (0.01, 0.03)},
}


# =======================================================
# [2. 데이터 로더]
# =======================================================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.map = config["CSV_COL_MAP"]

    def get_absolute_path(self, relative_path_from_csv):
        # CSV 값이 비어있거나 nan인 경우 처리
        if not relative_path_from_csv or str(relative_path_from_csv).lower() == 'nan':
            return None

        clean_rel = str(relative_path_from_csv).strip().replace('\\', '/')
        if clean_rel.startswith('./'):
            clean_rel = clean_rel[2:]
        elif clean_rel.startswith('/'):
            clean_rel = clean_rel[1:]

        abs_path = os.path.join(self.config["INPUT_ROOT"], clean_rel)
        return abs_path if os.path.exists(abs_path) else None

    def create_tasks(self):
        csv_path = os.path.join(self.config["INPUT_ROOT"], self.config["TARGET_CSV"])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Target CSV not found: {csv_path}")

        tasks = []
        print(f"📄 Reading target list from: {csv_path}")

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # 필수 컬럼(이미지 경로) 확인
            required = [self.map['REF_IMG'], self.map['TAR_IMG']]
            for req in required:
                if req not in headers:
                    print(f"❌ Error: CSV header '{req}' not found. Available: {headers}")
                    return []

            for row in reader:
                # 매핑된 컬럼명으로 데이터 읽기 (before는 읽지 않음)
                real_ref = self.get_absolute_path(row.get(self.map['REF_IMG'], ''))
                real_tar = self.get_absolute_path(row.get(self.map['TAR_IMG'], ''))
                real_ref_mask = self.get_absolute_path(row.get(self.map['REF_MASK'], ''))
                real_tar_mask = self.get_absolute_path(row.get(self.map['TAR_MASK'], ''))

                # 원본/타겟 이미지가 둘 다 있어야 작업 목록에 추가
                if real_ref and real_tar:
                    tasks.append({
                        'real_ref_path': real_ref,
                        'real_tar_path': real_tar,
                        'real_ref_mask_path': real_ref_mask,
                        'real_tar_mask_path': real_tar_mask,
                        'ref_filename': os.path.basename(real_ref),
                        'tar_filename': os.path.basename(real_tar)
                    })

                if self.config["IS_TEST_MODE"] and len(tasks) >= self.config["TEST_COUNT"]:
                    break
        return tasks


# =======================================================
# [3. 헬퍼 함수]
# =======================================================
def save_image_immediate(img, folder_path, filename):
    if img is None: return None
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    img.save(save_path, compress_level=1)
    return save_path


def is_black_image(pil_img, threshold=5.0):
    if pil_img is None: return True
    try:
        return np.mean(np.array(pil_img)) < threshold
    except Exception:
        return True


def load_image_with_retry(path, retries=3, delay=0.2, is_mask=False):
    if not path: return None
    for i in range(retries):
        try:
            with Image.open(path) as f:
                if is_mask:
                    img = f.convert("L")
                else:
                    img = f.convert("RGB")
                img.load()
                return img
        except Exception:
            if i == retries - 1: return None
            time.sleep(delay)
    return None


# =======================================================
# [4. 이미지 증강 엔진 (완벽 동기화 + 마스크 전체 적용)]
# =======================================================
class ImageAugmentor:
    @staticmethod
    def apply_noise(img, severity_factor):
        img_tensor = v2.ToImage()(img)
        img_tensor = v2.ToDtype(torch.float32, scale=True)(img_tensor)
        noise = torch.randn_like(img_tensor) * severity_factor
        noisy_img = torch.clamp(img_tensor + noise, 0., 1.)
        return v2.ToPILImage()(noisy_img), {"severity": round(severity_factor, 4)}

    @classmethod
    def apply_op(cls, img, tag, manual_param=None, is_mask=False):
        if img is None: return None, {}
        if tag not in PARAM_MAP: return img, {}

        config = PARAM_MAP[tag]
        method = config["method"]
        processed = img.copy()

        interp_mode = v2.InterpolationMode.NEAREST if is_mask else v2.InterpolationMode.BILINEAR
        fill_color = 0

        params_log = {}

        # manual_param(Seed) 우선 적용
        if method == "shear":
            min_v, max_v = config["range"]
            if manual_param:
                val, axis = manual_param
            else:
                val = random.uniform(min_v, max_v) * random.choice([-1, 1])
                axis = random.choice(['x', 'y'])

            processed = F.affine(processed, angle=0, translate=[0, 0], scale=1.0,
                                 shear=[val, 0.0] if axis == 'x' else [0.0, val],
                                 interpolation=interp_mode, fill=fill_color)
            params_log = {"axis": axis, "val": round(val, 2)}

        elif method == "rotate":
            min_v, max_v = config["range"]
            val = manual_param if manual_param is not None else random.uniform(min_v, max_v) * random.choice([-1, 1])
            processed = F.rotate(processed, angle=val,
                                 interpolation=interp_mode, fill=fill_color)
            params_log = {"angle": round(val, 2)}

        elif method == "hue":
            if processed.mode == 'L': processed = processed.convert('RGB')
            min_v, max_v = config["range"]
            val = manual_param if manual_param is not None else max(-0.5,
                                                                    min(random.uniform(min_v, max_v) * random.choice(
                                                                        [-1, 1]), 0.5))
            processed = F.adjust_hue(processed, val)
            if is_mask: processed = processed.convert('L')
            params_log = {"hue_factor": round(val, 3)}

        elif method == "grayscale":
            alpha = manual_param if manual_param is not None else random.uniform(*config["range"])
            if processed.mode != 'RGB': processed = processed.convert('RGB')
            processed = Image.blend(processed, ImageOps.grayscale(processed).convert("RGB"), alpha)
            if is_mask: processed = processed.convert('L')
            params_log = {"gray_alpha": round(alpha, 2)}

        elif method == "brightness":
            min_v, max_v = config["range"]
            factor = manual_param if manual_param is not None else max(0.0, 1.0 + (
                        random.uniform(min_v, max_v) * random.choice([-1, 1])))
            processed = F.adjust_brightness(processed, factor)
            params_log = {"bright_factor": round(factor, 2)}

        elif method == "contrast":
            val = manual_param if manual_param is not None else random.uniform(*config["range"])
            processed = F.adjust_contrast(processed, val)
            params_log = {"contrast_factor": round(val, 2)}

        elif method == "equalize":
            alpha = manual_param if manual_param is not None else random.uniform(*config["range"])
            if processed.mode != 'RGB': processed = processed.convert('RGB')
            processed = Image.blend(processed, ImageOps.equalize(processed), alpha)
            if is_mask: processed = processed.convert('L')
            params_log = {"eq_alpha": round(alpha, 2)}

        elif method == "noise":
            val = manual_param if manual_param is not None else random.uniform(*config["range"])
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
        elif method == "brightness":
            min_v, max_v = config["range"]
            return max(0.0, 1.0 + (random.uniform(min_v, max_v) * random.choice([-1, 1])))
        elif method == "contrast":
            return random.uniform(*config["range"])
        elif method == "equalize":
            return random.uniform(*config["range"])
        elif method == "noise":
            return random.uniform(*config["range"])

        return None


# =======================================================
# [5. 데이터 객체]
# =======================================================
class AugData:
    def __init__(self, ref_path, tgt_path, ref_mask_path, tgt_mask_path,
                 ref_name, tgt_name,
                 aug_method=None, aug_params=None,
                 img_ref_obj=None, img_tgt_obj=None,
                 mask_ref_obj=None, mask_tgt_obj=None):

        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.ref_mask_path = ref_mask_path
        self.tgt_mask_path = tgt_mask_path

        self.ref_name = ref_name
        self.tgt_name = tgt_name

        self.aug_method = aug_method if aug_method else ""
        self.aug_params = aug_params if aug_params else "{}"

        self.img_ref_obj = img_ref_obj
        self.img_tgt_obj = img_tgt_obj
        self.mask_ref_obj = mask_ref_obj
        self.mask_tgt_obj = mask_tgt_obj

    def get_images(self):
        if self.img_ref_obj is None: self.img_ref_obj = load_image_with_retry(self.ref_path)
        if self.img_tgt_obj is None: self.img_tgt_obj = load_image_with_retry(self.tgt_path)

        if self.mask_ref_obj is None and self.ref_mask_path:
            self.mask_ref_obj = load_image_with_retry(self.ref_mask_path, is_mask=True)
        if self.mask_tgt_obj is None and self.tgt_mask_path:
            self.mask_tgt_obj = load_image_with_retry(self.tgt_mask_path, is_mask=True)

        return self.img_ref_obj, self.img_tgt_obj, self.mask_ref_obj, self.mask_tgt_obj

    def release_memory(self):
        self.img_ref_obj = None
        self.img_tgt_obj = None
        self.mask_ref_obj = None
        self.mask_tgt_obj = None

    def get_target_subfolder(self):
        return "augmented"

    def get_csv_row(self):
        def to_relative(path):
            if not path: return ""
            try:
                rel = os.path.relpath(path, CONFIG["OUTPUT_ROOT"]).replace(os.sep, '/')
                return rel if rel.startswith('./') else './' + rel
            except:
                return path

        return {
            'tar_path': to_relative(self.tgt_path),
            'ref_path': to_relative(self.ref_path),
            'tar_mask_path': to_relative(self.tgt_mask_path),
            'ref_mask_path': to_relative(self.ref_mask_path),
            'aug_method': self.aug_method,
            'aug_param': self.aug_params
        }


# =======================================================
# [6. 워커 프로세스]
# =======================================================
def worker_process(task_data):
    try:
        real_ref_path = task_data['real_ref_path']
        real_tgt_path = task_data['real_tar_path']
        real_ref_mask = task_data.get('real_ref_mask_path')
        real_tgt_mask = task_data.get('real_tar_mask_path')

        ref_filename = task_data['ref_filename']
        tar_filename = task_data['tar_filename']

        base = os.path.basename(ref_filename)
        base_ref_name = base[:-8] if base.lower().endswith('.png.jpg') else os.path.splitext(base)[0]

        stem = os.path.basename(tar_filename)
        stem = stem[:-8] if stem.lower().endswith('.png.jpg') else os.path.splitext(stem)[0]
        base_tgt_name = stem.split("@seg")[0] + "@seg" if "@seg" in stem else stem

        origin_ref_img = load_image_with_retry(real_ref_path)
        origin_tgt_img = load_image_with_retry(real_tgt_path)
        origin_ref_mask = load_image_with_retry(real_ref_mask, is_mask=True)
        origin_tgt_mask = load_image_with_retry(real_tgt_mask, is_mask=True)

        if not origin_ref_img or not origin_tgt_img: return []

        pool = [AugData(real_ref_path, real_tgt_path, real_ref_mask, real_tgt_mask,
                        base_ref_name, base_tgt_name,
                        img_ref_obj=origin_ref_img, img_tgt_obj=origin_tgt_img,
                        mask_ref_obj=origin_ref_mask, mask_tgt_obj=origin_tgt_mask)]

        result_rows = []

        for step_idx, (step_num, scope, methods) in enumerate(AUG_STEPS):
            new_items = []
            if len(pool) > CONFIG["MAX_RAM_CACHE"]:
                for old_item in pool[:-CONFIG["MAX_RAM_CACHE"]]: old_item.release_memory()
                gc.collect()

            for data in pool:
                src_ref_img, src_tgt_img, src_ref_mask, src_tgt_mask = data.get_images()
                if not src_ref_img or not src_tgt_img: continue

                for tag in methods:
                    retry_count = 0
                    success = False
                    res_ref_img, res_tgt_img = None, None
                    res_ref_mask, res_tgt_mask = None, None
                    params_r, params_t = {}, {}

                    try:
                        history = json.loads(data.aug_params)
                    except:
                        history = {}

                    current_aug_method_str = data.aug_method

                    while retry_count < 3:
                        # [핵심] 하나의 Seed 생성 -> 4장 모두 동일하게 적용
                        seed_param = ImageAugmentor.generate_seed_param(tag)

                        if scope == 'pair':
                            # Pair: Ref/Tgt 모두 동일 Seed 적용
                            res_ref_img, params_r = ImageAugmentor.apply_op(src_ref_img, tag, seed_param)
                            res_ref_mask, _ = ImageAugmentor.apply_op(src_ref_mask, tag, seed_param, is_mask=True)

                            res_tgt_img, params_t = ImageAugmentor.apply_op(src_tgt_img, tag, seed_param)
                            res_tgt_mask, _ = ImageAugmentor.apply_op(src_tgt_mask, tag, seed_param, is_mask=True)

                            name_r = f"{data.ref_name}@{tag}"
                            name_t = f"{data.tgt_name}@{tag}"

                        else:  # tgt_only
                            # Ref 유지
                            res_ref_img = src_ref_img
                            res_ref_mask = src_ref_mask
                            params_r = None

                            # Tgt: Img + Mask 동일 Seed 적용
                            res_tgt_img, params_t = ImageAugmentor.apply_op(src_tgt_img, tag, seed_param)
                            res_tgt_mask, _ = ImageAugmentor.apply_op(src_tgt_mask, tag, seed_param, is_mask=True)

                            name_r = data.ref_name
                            name_t = f"{data.tgt_name}@{tag}"

                        if is_black_image(res_ref_img) or is_black_image(res_tgt_img):
                            retry_count += 1
                            continue
                        else:
                            success = True
                            break

                    if not success: continue

                    step_key = f"{step_num}_{tag}"
                    history[step_key] = {
                        "ref": params_r,
                        "tgt": params_t
                    }
                    aug_info_json = json.dumps(history, ensure_ascii=False)
                    new_method_str = f"{current_aug_method_str} > {tag}" if current_aug_method_str else tag

                    temp_item = AugData(None, None, None, None, name_r, name_t)
                    target_subfolder = temp_item.get_target_subfolder()

                    save_dir_ref = os.path.join(CONFIG["OUTPUT_ROOT"], "ref_img")
                    save_dir_ref_mask = os.path.join(CONFIG["OUTPUT_ROOT"], "ref_mask")
                    save_dir_tgt = os.path.join(CONFIG["OUTPUT_ROOT"], "tar_img", target_subfolder)
                    save_dir_tgt_mask = os.path.join(CONFIG["OUTPUT_ROOT"], "tar_mask", target_subfolder)

                    saved_ref_path = save_image_immediate(res_ref_img, save_dir_ref, f"{name_r}.png")
                    saved_tgt_path = save_image_immediate(res_tgt_img, save_dir_tgt, f"{name_t}.png")

                    saved_ref_mask_path = None
                    if res_ref_mask:
                        saved_ref_mask_path = save_image_immediate(res_ref_mask, save_dir_ref_mask, f"{name_r}.png")

                    saved_tgt_mask_path = None
                    if res_tgt_mask:
                        saved_tgt_mask_path = save_image_immediate(res_tgt_mask, save_dir_tgt_mask, f"{name_t}.png")

                    next_item = AugData(saved_ref_path, saved_tgt_path, saved_ref_mask_path, saved_tgt_mask_path,
                                        name_r, name_t,
                                        aug_method=new_method_str,
                                        aug_params=aug_info_json,
                                        img_ref_obj=res_ref_img, img_tgt_obj=res_tgt_img,
                                        mask_ref_obj=res_ref_mask, mask_tgt_obj=res_tgt_mask)

                    result_rows.append(next_item.get_csv_row())
                    new_items.append(next_item)

            pool.extend(new_items)

        del pool
        gc.collect()
        return result_rows

    except Exception:
        print(f"\n[Error processing {task_data.get('tar_filename')}]")
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

    loader = DataLoader(CONFIG)
    try:
        all_tasks = loader.create_tasks()
    except Exception as e:
        print(f"❌ Error creating tasks: {e}")
        return

    task_list = [t for t in all_tasks if t['tar_filename'] not in completed_tasks]
    if not task_list:
        print("✅ All tasks completed!")
        return

    print(f"🚀 Start (Workers: {CONFIG['NUM_WORKERS']}) - Remaining: {len(task_list)}")

    headers = ['tar_path', 'ref_path', 'tar_mask_path', 'ref_mask_path', 'aug_method', 'aug_param']

    file_mode = 'a' if (completed_tasks and os.path.exists(output_csv)) else 'w'

    with open(output_csv, file_mode, newline='', encoding='utf-8-sig') as out_f, \
            open(checkpoint_path, 'a', encoding='utf-8') as cp_f:

        writer = csv.DictWriter(out_f, fieldnames=headers)
        if file_mode == 'w': writer.writeheader()

        with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
            future_to_id = {executor.submit(worker_process, task): task['tar_filename'] for task in task_list}

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

    print(f"✅ Finished: {CONFIG['OUTPUT_ROOT']}")


if __name__ == "__main__":
    main()