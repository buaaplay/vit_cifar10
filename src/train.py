import argparse
import sys
import traceback
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import timm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast

import logging

def parse_args():
    parser = argparse.ArgumentParser(description='ViT CIFAR10 Training')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to datasets')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay')  # 调低权重衰减
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=10, help='Patience for Early Stopping')  # Early Stopping 的耐心轮数
    args = parser.parse_args()

    # 打印接收到的参数
    print(f"Received arguments: {sys.argv}", flush=True)

    return args

def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return local_rank, world_size, rank

def get_dataloaders(args, world_size, rank):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_val)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=16, pin_memory=True)

    return train_loader, val_loader

def build_model():
    # 使用 ViT-base Patch16 224 模型，启用 Dropout
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10, drop_rate=0.1)
    return model

def generate_log_dir(args):
    # 生成基于参数和当前时间的日志目录名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/batch{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_modelvit_base_patch16_224_{timestamp}"
    return log_dir

def setup_logging(log_file, rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    if rank == 0 and log_file:
        # 主进程写入文件
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 所有进程输出到标准输出
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def train_epoch(epoch, model, criterion, optimizer, train_loader, device, rank, writer, scaler, accumulation_steps, logger):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    if writer and rank == 0:
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Train/Epoch_Time', epoch_duration, epoch)

    # 记录日志（仅 rank 0）
    if rank == 0:
        logger.info(f"Epoch [{epoch}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Train Time: {epoch_duration:.2f}s")

    return epoch_loss, epoch_acc, epoch_duration

def validate(epoch, model, criterion, val_loader, device, rank, writer, logger):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    if writer and rank == 0:
        writer.add_scalar('Validation/Loss', epoch_loss, epoch)
        writer.add_scalar('Validation/Accuracy', epoch_acc, epoch)
        writer.add_scalar('Validation/Epoch_Time', epoch_duration, epoch)

    # 记录日志（仅 rank 0）
    if rank == 0:
        logger.info(f"Epoch [{epoch}] Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}% | Val Time: {epoch_duration:.2f}s")

    return epoch_loss, epoch_acc, epoch_duration

def save_checkpoint(state, checkpoint_dir, epoch, rank):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_rank_{rank}.pth')
    torch.save(state, filename)

def main():
    try:
        args = parse_args()

        # 指定模型名称
        args.model_name = 'vit_base_patch16_224'

        local_rank, world_size, rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        # 生成基于参数的日志目录
        log_dir = generate_log_dir(args)

        # 生成基于时间和参数的详细日志文件名（仅 rank 0）
        if rank == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"./logs/batch{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_model{args.model_name}_{timestamp}.log"
        else:
            log_file = None

        logger = setup_logging(log_file, rank)

        if rank == 0:
            logger.info("开始执行训练脚本...")
            logger.info(f"日志将保存到 {log_dir}")
            logger.info(f"训练参数: {vars(args)}")  # 记录所有训练参数
            logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 初始化 TensorBoard（仅 rank 0）
        if rank == 0:
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        logger.info("加载数据...")
        train_loader, test_loader = get_dataloaders(args, world_size, rank)
        logger.info("构建模型...")
        model = build_model().to(device)

        if world_size > 1:
            logger.info("转换为 SyncBatchNorm 并使用 DistributedDataParallel...")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        logger.info("设置损失函数和优化器...")
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # 初始化混合精度训练的梯度缩放器
        scaler = GradScaler()

        best_acc = 0.0
        total_start_time = time.time()

        # Early Stopping 参数
        patience = args.patience
        trigger_times = 0

        for epoch in range(1, args.epochs + 1):
            if rank == 0:
                logger.info(f"开始第 {epoch} 轮训练。")
            train_loss, train_acc, train_time = train_epoch(
                epoch, model, criterion, optimizer, train_loader, device, rank, writer, scaler, args.accumulation_steps, logger
            )
            val_loss, val_acc, val_time = validate(
                epoch, model, criterion, test_loader, device, rank, writer, logger
            )
            scheduler.step()

            if rank == 0:
                total_elapsed_time = time.time() - total_start_time
                logger.info(f"Epoch [{epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Time: {train_time:.2f}s | "
                            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Time: {val_time:.2f}s | "
                            f"Total Elapsed Time: {total_elapsed_time:.2f}s")
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_acc,
                    }, args.checkpoint_dir, epoch, rank)
                    trigger_times = 0
                else:
                    trigger_times += 1
                    logger.info(f"Validation Acc did not improve for {trigger_times} epochs.")
                    if trigger_times >= patience:
                        logger.info("Early stopping triggered.")
                        break

        if rank == 0 and writer:
            writer.close()

        dist.destroy_process_group()
        logger.info("训练脚本已完成。")
    except Exception as e:
        # 仅 rank 0 记录错误日志
        if 'logger' in locals() and rank == 0:
            logger.error(f"训练过程中出现错误: {e}")
        else:
            print(f"训练过程中出现错误: {e}")
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except:
            pass
        sys.exit(1)

if __name__ == '__main__':
    main()
