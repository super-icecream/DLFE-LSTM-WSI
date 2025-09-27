#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练监控脚本
实时监控训练进度和性能指标
"""

import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def monitor_logs(log_dir: Path, refresh_interval: int = 5):
    """
    监控训练日志
    
    Args:
        log_dir: 日志目录
        refresh_interval: 刷新间隔(秒)
    """
    print("="*60)
    print("训练监控")
    print("="*60)
    print(f"日志目录: {log_dir}")
    print(f"刷新间隔: {refresh_interval}秒")
    print("按Ctrl+C停止监控")
    print("="*60)
    
    # 查找最新的日志文件
    log_files = list(log_dir.glob('*/experiment.log'))
    if not log_files:
        print("未找到日志文件")
        return
    
    latest_log = max(log_files, key=os.path.getctime)
    print(f"监控日志: {latest_log}")
    
    # 持续监控
    last_position = 0
    metrics_history = []
    
    try:
        while True:
            # 读取新行
            with open(latest_log, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
            
            # 解析并显示
            for line in new_lines:
                # 提取关键信息
                if 'Epoch' in line and 'Loss' in line:
                    print(line.strip())
                    
                    # 尝试提取数值
                    try:
                        parts = line.split('|')
                        for part in parts:
                            if 'Loss' in part:
                                loss = float(part.split(':')[-1].strip())
                                metrics_history.append({
                                    'time': datetime.now(),
                                    'loss': loss
                                })
                    except:
                        pass
                elif 'RMSE' in line or 'MAE' in line:
                    print(line.strip())
            
            # 更新可视化
            if len(metrics_history) > 1:
                update_visualization(metrics_history)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n监控停止")
        
        # 保存历史
        if metrics_history:
            save_metrics_history(metrics_history, log_dir)


def update_visualization(metrics_history: list):
    """更新可视化"""
    # 简单的进度显示
    if metrics_history:
        latest = metrics_history[-1]
        print(f"\r最新Loss: {latest['loss']:.6f} | "
              f"时间: {latest['time'].strftime('%H:%M:%S')}", 
              end='', flush=True)


def save_metrics_history(metrics_history: list, output_dir: Path):
    """保存指标历史"""
    df = pd.DataFrame(metrics_history)
    
    # 保存CSV
    csv_path = output_dir / 'metrics_history.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n指标历史保存至: {csv_path}")
    
    # 生成图表
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(df)), df['loss'], 'b-', label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = output_dir / 'training_progress.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"训练曲线保存至: {plot_path}")


def launch_tensorboard(log_dir: Path):
    """启动TensorBoard"""
    import subprocess
    
    print(f"启动TensorBoard...")
    print(f"日志目录: {log_dir}")
    
    try:
        process = subprocess.Popen(
            ['tensorboard', '--logdir', str(log_dir), '--port', '6006'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("TensorBoard已启动")
        print("访问: http://localhost:6006")
        print("按Ctrl+C停止")
        
        process.wait()
        
    except Exception as e:
        print(f"TensorBoard启动失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='训练监控脚本')
    parser.add_argument('--log-dir', type=str, 
                       default='./experiments/logs',
                       help='日志目录')
    parser.add_argument('--interval', type=int, default=5,
                       help='刷新间隔(秒)')
    parser.add_argument('--tensorboard', action='store_true',
                       help='启动TensorBoard')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if args.tensorboard:
        launch_tensorboard(log_dir)
    else:
        monitor_logs(log_dir, args.interval)


if __name__ == '__main__':
    main()