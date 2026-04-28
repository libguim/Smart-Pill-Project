import codecs, json
import numpy as np
from PIL import Image, ImageOps
import cv2
import torch
import os
import time
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import logging


def inverse_vgg_preprocess(image):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    image = image.copy().transpose((1, 2, 0))
    for i in range(3):
        image[:, :, i] = image[:, :, i] * stds[i]
        image[:, :, i] = image[:, :, i] + means[i]
    image = image[:, :, ::-1]
    image = image * 255
    image[image > 255.] = 255.
    image[image < 0.] = 0.
    image = image.astype(np.uint8)
    return image


def _make_ocr_image(crop_bgr: np.ndarray) -> list:
    """
    OCR 전용 다중 전처리 이미지 생성. (필루미 벤치마킹 포함)

    저대비 알약 각인은 단일 전처리로 OCR이 텍스트 영역을 감지하지 못합니다.
    5가지 전처리 버전을 리스트로 반환하고, _run_ocr에서 모두 시도해
    가장 긴 결과를 채택합니다.

    v1~v4: 기존 전처리 (CLAHE, 감마 보정 등)
    v5:    필루미 벤치마킹 — equalizeHist + fastNlMeansDenoising
           (필루미 Model.py의 각인 전처리 파이프라인 적용)

    반환: [img1, img2, img3, img4, img5] — 모두 numpy ndarray (그레이스케일)
    """
    # 3배 업스케일 (각인이 작을수록 OCR 정확도 향상)
    upscaled = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # ── v1: CLAHE + 샤프닝 (기본) ──────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    v1 = cv2.filter2D(clahe.apply(gray), -1, sharp_kernel)

    # ── v2: 강한 CLAHE (저대비 각인용) ─────────────────
    clahe2 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(2, 2))
    v2 = clahe2.apply(gray)

    # ── v3: 히스토그램 평활화 + 샤프닝 ─────────────────
    v3 = cv2.filter2D(cv2.equalizeHist(gray), -1, sharp_kernel)

    # ── v4: 역감마 + 샤프닝 (양각 각인 강조) ───────────
    lut_inv = np.array([((i / 255.0) ** 2.0) * 255 for i in range(256)], dtype=np.uint8)
    v4 = cv2.filter2D(cv2.LUT(gray, lut_inv), -1, sharp_kernel)

    # ── v5: 필루미 방식 — equalizeHist + fastNlMeansDenoising ──
    # 필루미 Model.py pill_classification_top5의 각인 전처리 파이프라인:
    #   gray → equalizeHist → fastNlMeansDenoising(h=10, 7, 21) → /255
    # OCR용이므로 정규화(/255)는 생략하고 uint8 그대로 반환.
    eq = cv2.equalizeHist(gray)
    v5 = cv2.fastNlMeansDenoising(eq, None, h=10, templateWindowSize=7, searchWindowSize=21)

    return [v1, v2, v3, v4, v5]


def preprocess_for_inference(image_obj):
    """
    Django 이미지 객체를 입력받아 중앙의 알약을 정밀 검출하고,
    CNN용 PIL Image, OCR용 다중 전처리 리스트, 색상감지용 원본 크롭 PIL을 반환합니다.

    반환: (PIL Image for CNN, list[ndarray] for OCR, PIL Image for 색상감지)
          알약 윤곽선 미검출 시 (None, None, None)
    """
    try:
        # 파일 확장자 유효성 체크
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        ext = os.path.splitext(image_obj.name)[1].lower()
        if ext not in valid_extensions:
            print(f"🚫 [VALIDATION_ERROR] 지원하지 않는 파일 형식: {ext}")
            return None, None, None                          # ← 수정 1

        # EXIF 보정 후 OpenCV BGR 배열로 직접 변환 (이중 로드 없음)
        image_obj.seek(0)
        temp_img = Image.open(image_obj)
        temp_img = ImageOps.exif_transpose(temp_img)
        img = cv2.cvtColor(np.array(temp_img.convert('RGB')), cv2.COLOR_RGB2BGR)

        if img is None:
            print(f"⚠️ 이미지 디코딩 실패")
            return None, None, None                          # ← 수정 2

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(current_file_dir, 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)

        img_h, img_w = img.shape[:2]
        img_center = (img_w // 2, img_h // 2)

        # 1. 대비 강화 및 블러 처리
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        blurred = cv2.GaussianBlur(clahe_img, (11, 11), 0)

        # 2. 엣지 검출 및 모폴로지
        edged = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # 3. 윤곽선 검출 및 후보군 필터링
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Canny 실패 시 Otsu 이진화로 재시도 (흰 배경 + 저대비 알약 대응)
        # 정답을 잘 맞추던 알약은 Canny가 성공하므로 이 블록이 실행되지 않음
        if max((cv2.contourArea(c) for c in contours), default=0) < 800:
            print(f"[PREPROCESS] Canny 윤곽선 부족 → Otsu 이진화로 재시도")
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            otsu_mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
            otsu_contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if max((cv2.contourArea(c) for c in otsu_contours), default=0) >= 800:
                contours = otsu_contours
                print(f"[PREPROCESS] Otsu 재시도 성공: {len(contours)}개 윤곽선")

        pill_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:
                    pill_candidates.append(cnt)

        # 4. 중앙 우선순위로 최적 객체 선택
        best_pill = None
        min_dist_to_center = float('inf')

        for cnt in pill_candidates:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dist = np.sqrt((cX - img_center[0])**2 + (cY - img_center[1])**2)
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                best_pill = cnt

        # 5. 알약 미검출 → 추론 중단 (무턱대고 중앙 크롭하지 않음)
        if best_pill is None:
            print(f"⚠️ [PREPROCESS] 알약 윤곽선 미검출 → 추론 중단")
            timestamp = int(time.time() * 1000)
            debug_filename = f"pill_UNDETECTED_{timestamp}.png"
            try:
                debug_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_img)
                print(f"📸 [UNDETECTED] 전처리 이미지 저장 완료: {debug_filename}")
            except Exception:
                pass
            return None, None, None                          # ← 수정 3

        # 6. 알약 크롭 (마진 포함)
        x, y, w, h = cv2.boundingRect(best_pill)
        margin_w, margin_h = int(w * 0.40), int(h * 0.40)
        y1 = max(0, y - margin_h)
        y2 = min(img_h, y + h + margin_h)
        x1 = max(0, x - margin_w)
        x2 = min(img_w, x + w + margin_w)
        crop_img = img[y1:y2, x1:x2]

        # 7. OCR 전용 다중 전처리 이미지 생성 (크롭된 원본 기준)
        ocr_img = _make_ocr_image(crop_img)

        # 7-1. 색상 감지용 원본 크롭 PIL (정규화 전 BGR → RGB 변환)
        color_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

        # 8. CNN용: 정사각형 캔버스 배치 및 224x224 리사이즈
        ch, cw = crop_img.shape[:2]
        size = max(ch, cw)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[(size-ch)//2:(size-ch)//2+ch, (size-cw)//2:(size-cw)//2+cw] = crop_img
        final_img = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA)

        # 9. 디버깅용 이미지 저장 (CNN용 + OCR 4버전)
        timestamp = int(time.time() * 1000)
        debug_cnn = f"pill_DETECTED_{timestamp}.png"
        try:
            cv2.imwrite(os.path.join(debug_dir, debug_cnn), final_img)
            # OCR 4버전을 가로로 이어붙여 한 장으로 저장
            ocr_saves = [cv2.resize(v, (224, 224), interpolation=cv2.INTER_AREA) for v in ocr_img]
            ocr_combined = np.hstack(ocr_saves)
            debug_ocr = f"pill_OCR_{timestamp}.png"
            cv2.imwrite(os.path.join(debug_dir, debug_ocr), ocr_combined)
            print(f"📸 [DETECTED] CNN: {debug_cnn} / OCR(4버전): {debug_ocr}")
        except Exception as e:
            print(f"⚠️ 이미지 저장 실패: {e}")

        # 10. 반환: CNN용 PIL + OCR용 리스트 + 색상감지용 원본크롭 PIL
        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_img_rgb), ocr_img, color_pil  # ← 수정 4

    except Exception as e:
        print(f"❌ 전처리 중 서버 에러 발생: {e}")
        return None, None


##########################################################################
def save_dict_to_json(dict_save, filejson, mode='w'):
    with codecs.open(filejson, mode, encoding='utf-8') as f:
        json.dump(dict_save, f, ensure_ascii=False, indent=4)


def read_dict_from_json(filejson):
    if not os.path.isfile(filejson):
        return None
    with codecs.open(filejson, 'r', encoding='utf-8') as f:
        obj = json.load(f)
        return obj

def open_opencv_file(filename):
    img_array = np.fromfile(filename, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

def save_opencv_file(image, filename):
    result, encoded_img = cv2.imencode('.jpg', image)
    if result:
        with open(filename, mode='w+b') as f:
            encoded_img.tofile(f)
            return True
    else:
        return False

def show_cvimage(image):
    cv2.imshow('a', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show_tensor3(inp, cmap=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)
    plt.show()
    plt.close()

def save_tensor3(inp, filename):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    matplotlib.image.imsave(filename, inp)

def model_save(model_path, epoch, model, optimizer, rank=0):
    if rank == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer != None else 0,
        }, model_path)
        print(f'model was saved to {model_path}')

def model_load(args, model, optimizer, rank=0):
    epoch_begin = 0
    state_model = None
    state_optimizer = None
    if not os.path.isfile(args.model_path_in):
        print(f"model_path doesn't exist:{args.model_path_in}")
        return epoch_begin, None, False
    if args.verbose == True: print(f"model_path will be loaded from:{args.model_path_in}")
    dict_checkpoint = torch.load(args.model_path_in, map_location='cpu')
    epoch_begin = dict_checkpoint.get('epoch', -1) + 1
    if rank == 0:
        try:
            state_model = dict_checkpoint.get('model', None)
            if state_model != None:
                if not hasattr(model, 'module') and ('module' in list(state_model.keys())[0]):
                    state_model = OrderedDict([(k[7:], v) if 'module' in k else (k, v) for k, v in state_model.items()])
                    model.load_state_dict(state_model)
                elif hasattr(model, 'module') and (not 'module' in list(state_model.keys())[0]):
                    state_model = OrderedDict([('module.' + k, v) for k, v in state_model.items()])
                    model.load_state_dict(state_model)
                else:
                    model.load_state_dict(state_model)
        except Exception as e:
            print(f'Fail to loading model: {e}')
            return epoch_begin, dict_checkpoint, False
        if optimizer != None and args.run_phase == 'train':
            try:
                state_optimizer = dict_checkpoint.get('optimizer', None)
                if state_optimizer != None:
                    optimizer.load_state_dict(state_optimizer)
            except Exception as e:
                print(f'Fail to loading optimizer: {e}')
                return epoch_begin, dict_checkpoint, False
    success = True if state_model != None and state_optimizer != None else False
    return epoch_begin, dict_checkpoint, success

def convert_pil_to_cv2(pil_image):
    pil_image = pil_image.convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def convert_cv2_to_pil(cv_image):
    return Image.fromarray(cv_image)

def save_img_paf_heat(path, img_temp, paf_temp, heatmap_temp, args):
    path = Path(path)
    if path.parts[-2].isdigit():
        dir_paf = os.path.join(args.dir_review_paf_train, path.parts[-2])
        dir_heat = os.path.join(args.dir_review_heat_train, path.parts[-2])
        dir_img = os.path.join(args.dir_review_img_train, path.parts[-2])
    else:
        dir_paf = args.dir_output
        dir_heat = args.dir_output
        dir_img = args.dir_output
    os.makedirs(dir_paf, exist_ok=True)
    os.makedirs(dir_heat, exist_ok=True)
    os.makedirs(dir_img, exist_ok=True)
    file_base = path.name.split('.')[0]
    save_opencv_file(inverse_vgg_preprocess(img_temp), os.path.join(dir_img, file_base + '_base.jpg'))
    save_opencv_file(inverse_vgg_preprocess(paf_temp), os.path.join(dir_paf, file_base + '_paf.jpg'))
    save_opencv_file(inverse_vgg_preprocess(heatmap_temp), os.path.join(dir_heat, file_base + '_heat.jpg'))

def open_pil_as_stack_gray_np(filename):
    np_pil = np.array(Image.open(filename).convert('L'))
    return np.dstack([np_pil, np_pil, np_pil])

def open_pil_as_stack_color_np(filename):
    return np.array(Image.open(filename))

def save_np_pil_file(np_image, filename):
    Image.fromarray(np_image).save(filename)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def saveimage(sub, dir_digit, basename):
    sub.save(os.path.join(dir_digit, basename + '.jpg'))

def extractDigit_saveto(file_json, file_bmp, list_dir_digit=None):
    dict_bmp_info = read_dict_from_json(file_json)
    digitFractNo = int(dict_bmp_info['digitFractNo'])
    digitAllNo = int(dict_bmp_info['digitAllNo'])
    dataValue = int(dict_bmp_info['dataValue'] * 10 ** digitFractNo)
    digitRect = dict_bmp_info['digitRect']
    str_dataValue = f'{dataValue:0{digitAllNo}}'
    str_igmsGaugeDataId = dict_bmp_info['igmsGaugeDataId']
    if len(str_dataValue) != digitAllNo:
        if len(str_igmsGaugeDataId) == digitAllNo:
            str_dataValue = str_igmsGaugeDataId
        else:
            raise Exception("improper data format")
    list_digitRect = [[int(a) for a in aa.split(',')] for aa in digitRect.split('|')[1:digitAllNo+1]]
    img = Image.open(file_bmp)
    if img == None:
        return
    list_image = []
    for index in range(digitAllNo):
        x, y, width, height = list_digitRect[index]
        sub = img.crop((x, y, x + width, y + height))
        if list_dir_digit != None:
            saveimage(sub, list_dir_digit[int(str_dataValue[index])], os.path.basename(file_json).split('.')[0] + f'_{index}{str_dataValue[index]}')
        else:
            list_image.append(sub.convert('RGB'))
    if list_dir_digit == None:
        return list_image, [int(aa) for aa in str_dataValue], dict_bmp_info

def get_Image_Value_List_from_json(file_json):
    list_image, list_value, dict_json_info = extractDigit_saveto(file_json, os.path.splitext(file_json)[0] + '.bmp')
    return [(list_image[i], list_value[i], file_json) for i in range(len(list_image))]

transform_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_classifier = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transfrom_paf = transforms.Compose([
    transforms.Resize((368, 368)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class Dataset_valid(Dataset):
    def __init__(self, file_json, transform):
        if os.path.isfile(file_json) and file_json.split('.')[-1] == 'json':
            self.list_cv_label_path = get_Image_Value_List_from_json(file_json)
        elif os.path.isdir(file_json):
            self.list_cv_label_path = [(Image.open(aa).convert("RGB"), int(str(Path(aa).parts[-2])), aa) for aa in glob(file_json + r'/**/*.jpg')]
        self.transform = transform
        self.file_json = file_json

    def __len__(self):
        return len(self.list_cv_label_path)

    def __getitem__(self, idx):
        image, label, path = self.list_cv_label_path[idx]
        if self.transform != None:
            image = self.transform(image)
        return image, (label, path)

def get_optimizer(args, model):
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    return optimizer

def adjust_learning_rate(args, optimizer, epoch):
    if epoch in args.lr_schedule:
        args.lr *= args.lr_gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def val(self):
        return self.sum

def create_logging(file_log):
    path_file_log = Path(file_log)
    path_file_log.parent.mkdir(exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    streamingHandler = logging.StreamHandler()
    streamingHandler.setFormatter(logging.Formatter(u'%(message)s'))
    file_handler = logging.FileHandler(file_log)
    file_handler.setFormatter(logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s'))
    logger.addHandler(streamingHandler)
    logger.addHandler(file_handler)
    return logger

if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(current_file_dir, 'debug_images')
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
        print(f"✅ 테스트 실행: 디렉토리 생성 완료 -> {debug_dir}")
    else:
        print(f"ℹ️ 디렉토리가 이미 존재합니다: {debug_dir}")