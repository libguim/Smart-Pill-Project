# 실행: python plot_curves.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 기존 코드 import (수정 없음)
from get_cli_args import get_cli_args
from pill_classifier import Dataset_Pill, get_pill_model
from utils import model_load, model_save, accuracy, get_optimizer, transform_normalize, AverageMeter


# ── 색상 팔레트 ────────────────────────────────────────────────
C = dict(
    bg         = '#FAFAFA',
    panel      = '#FFFFFF',
    grid       = '#F0F0F0',
    text       = '#1A1A2E',
    axis       = '#CCCCCC',
    train_loss = '#5B7FD4',   # 인디고 블루
    valid_loss = '#A78BFA',   # 라벤더 퍼플
    train_acc  = '#34D399',   # 에메랄드 그린
    valid_acc  = '#FB923C',   # 소프트 오렌지
)


# ── 학습 1 epoch ───────────────────────────────────────────────
def run_one_epoch_train(args, dataloader, model, criterion, optimizer, epoch):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()

    with tqdm(total=len(dataloader), desc=f'[Train] Epoch {epoch:03d}', ncols=90) as t:
        for imgs, targets in dataloader:
            if args.cuda:
                imgs, targets = imgs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            prec1, _ = accuracy(outputs, targets, (1, 5))
            n = imgs.size(0)
            loss_meter.update(loss.item(), n)
            top1_meter.update(prec1[0].item(), n)
            t.set_postfix(loss=f'{loss_meter.avg:.4f}', top1=f'{top1_meter.avg:.2f}%')
            t.update(1)

    return loss_meter.avg, top1_meter.avg


# ── 검증 1 epoch ───────────────────────────────────────────────
def run_one_epoch_valid(args, dataloader, model, criterion, epoch):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f'[Valid] Epoch {epoch:03d}', ncols=90) as t:
            for imgs, targets, *_ in dataloader:  # valid: path_img, aug_name 포함
                if args.cuda:
                    imgs, targets = imgs.cuda(), targets.cuda()

                outputs = model(imgs)
                loss = criterion(outputs, targets)

                prec1, _ = accuracy(outputs, targets, (1, 5))
                n = imgs.size(0)
                loss_meter.update(loss.item(), n)
                top1_meter.update(prec1[0].item(), n)
                t.set_postfix(loss=f'{loss_meter.avg:.4f}', top1=f'{top1_meter.avg:.2f}%')
                t.update(1)

    return loss_meter.avg, top1_meter.avg


# ── 축 스타일 공통 적용 ────────────────────────────────────────
def _style_ax(ax, ylabel):
    ax.set_facecolor(C['panel'])
    ax.set_xlabel('Epoch', color=C['text'], fontsize=10)
    ax.set_ylabel(ylabel, color=C['text'], fontsize=10)
    ax.tick_params(colors='#666666', labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_edgecolor(C['axis'])
        spine.set_linewidth(0.8)
    ax.grid(color=C['grid'], linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)


# ── 그래프 PNG 저장 (에포크마다 덮어쓰기) ─────────────────────
def save_training_curves(history, save_path):
    epochs     = history['epochs']
    train_loss = history['train_loss']
    valid_loss = history['valid_loss']
    train_top1 = history['train_top1']
    valid_top1 = history['valid_top1']

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5), facecolor=C['bg'])
    fig.suptitle('ResNet152  Training Curves', color=C['text'], fontsize=14, fontweight='bold', y=1.02)

    # Loss 패널
    _style_ax(ax_loss, 'Loss')
    ax_loss.plot(epochs, train_loss, color=C['train_loss'], lw=2.2, label='Train Loss', alpha=0.9)
    ax_loss.plot(epochs, valid_loss, color=C['valid_loss'], lw=2.2, linestyle='--', label='Valid Loss', alpha=0.9)
    ax_loss.fill_between(epochs, train_loss, alpha=0.07, color=C['train_loss'])
    ax_loss.fill_between(epochs, valid_loss, alpha=0.07, color=C['valid_loss'])

    # 최소 valid loss 지점 표시
    best_ep  = epochs[int(np.argmin(valid_loss))]
    best_val = min(valid_loss)
    ax_loss.axvline(best_ep, color=C['valid_loss'], lw=1.0, linestyle=':', alpha=0.6)
    ax_loss.scatter([best_ep], [best_val], color=C['valid_loss'], s=60, zorder=5)
    ax_loss.annotate(f'Best  {best_val:.4f}\nEpoch {best_ep}',
                     xy=(best_ep, best_val),
                     xytext=(best_ep + max(len(epochs) * 0.08, 1), best_val * 1.08),
                     color=C['valid_loss'], fontsize=8, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=C['valid_loss'], lw=1.2))
    ax_loss.legend(facecolor=C['panel'], edgecolor=C['axis'], fontsize=9, framealpha=1, loc='upper right')

    # Accuracy 패널
    _style_ax(ax_acc, 'Top-1 Accuracy (%)')
    ax_acc.set_ylim(0, 105)
    ax_acc.plot(epochs, train_top1, color=C['train_acc'], lw=2.2, label='Train Acc', alpha=0.9)
    ax_acc.plot(epochs, valid_top1, color=C['valid_acc'], lw=2.2, linestyle='--', label='Valid Acc', alpha=0.9)
    ax_acc.fill_between(epochs, train_top1, alpha=0.07, color=C['train_acc'])
    ax_acc.fill_between(epochs, valid_top1, alpha=0.07, color=C['valid_acc'])

    # 최고 valid acc 지점 표시
    best_ep2 = epochs[int(np.argmax(valid_top1))]
    best_acc = max(valid_top1)
    ax_acc.axvline(best_ep2, color=C['valid_acc'], lw=1.0, linestyle=':', alpha=0.6)
    ax_acc.scatter([best_ep2], [best_acc], color=C['valid_acc'], s=60, zorder=5)
    ax_acc.annotate(f'Best  {best_acc:.2f}%\nEpoch {best_ep2}',
                    xy=(best_ep2, best_acc),
                    xytext=(best_ep2 + max(len(epochs) * 0.08, 1), best_acc - 12),
                    color=C['valid_acc'], fontsize=8, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C['valid_acc'], lw=1.2))
    ax_acc.legend(facecolor=C['panel'], edgecolor=C['axis'], fontsize=9, framealpha=1, loc='lower right')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f'그래프 저장 -> {save_path}')


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':

    job  = 'resnet152'
    args = get_cli_args(job=job, run_phase='train', aug_level=1, dataclass='0')
    args.cuda = torch.cuda.is_available()

    # 데이터셋 로드
    print('데이터셋 로딩 중...')
    dataset_train = Dataset_Pill(args, args.json_pill_class_list, transform=transform_normalize, run_phase='train')
    dataset_valid = Dataset_Pill(args, args.json_pill_class_list, transform=transform_normalize, run_phase='valid')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'  train: {len(dataset_train)}장  /  valid: {len(dataset_valid)}장')

    # 모델 / 손실함수 / 옵티마이저
    cudnn.benchmark = True
    args.gpu  = 0 if args.cuda else None
    args.rank = args.gpu
    model     = get_pill_model(args)
    criterion = nn.CrossEntropyLoss().cuda() if args.cuda else nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    epoch_begin, _, _ = model_load(args, model, optimizer)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2,
                                 threshold=1e-4, cooldown=3, min_lr=0)

    # 학습 기록 버퍼
    history = dict(epochs=[], train_loss=[], valid_loss=[], train_top1=[], valid_top1=[])
    best_valid_loss = float('inf')
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_png = os.path.join(args.dir_log, f'training_curves_{ts}.png')

    # 학습 루프
    print(f'\n학습 시작 (epoch {epoch_begin} -> {args.epochs - 1})\n')
    for epoch in range(epoch_begin, args.epochs):

        t_loss, t_top1 = run_one_epoch_train(args, dataloader_train, model, criterion, optimizer, epoch)

        # valid는 10 epoch 이후부터 (기존 run_model 로직과 동일)
        if epoch > 10:
            v_loss, v_top1 = run_one_epoch_valid(args, dataloader_valid, model, criterion, epoch)
        else:
            v_loss, v_top1 = None, None

        history['epochs'].append(epoch)
        history['train_loss'].append(t_loss)
        history['valid_loss'].append(v_loss if v_loss is not None else t_loss)
        history['train_top1'].append(t_top1)
        history['valid_top1'].append(v_top1 if v_top1 is not None else t_top1)

        print(f'Epoch {epoch:03d} | Train Loss {t_loss:.4f}  Top1 {t_top1:.2f}% | '
              f'Valid Loss {v_loss if v_loss else "-":>8}  Top1 {v_top1 if v_top1 else "-"}')

        # valid loss 개선 시 모델 저장
        if v_loss is not None and v_loss < best_valid_loss:
            model_save(args.model_path, epoch, model, optimizer, rank=0)
            best_valid_loss = v_loss

        if v_loss is not None:
            lr_scheduler.step(v_loss)

        # 에포크마다 그래프 갱신
        save_training_curves(history, save_png)

    print(f'\n학습 완료!  최종 그래프: {save_png}')