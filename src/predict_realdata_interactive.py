import os
import sys
import torch
import numpy as np
import csv
from tqdm import tqdm

# Fix import for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict_realdata import find_image_pairs, load_ensemble_models, predict_with_ensemble
import config as Config

# Add CHANNELS_TO_USE here to avoid config.py dependency
if not hasattr(Config, 'CHANNELS_TO_USE'):
    Config.CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask']
# Add INPUT_CHANNELS to Config to match predict_realdata.py
if not hasattr(Config, 'INPUT_CHANNELS'):
    def _calculate_input_channels(channels_list):
        count = 0
        if 'rgb' in channels_list: count += 3
        if 'gray' in channels_list: count += 1
        if 'hsv' in channels_list: count += 3
        if 'lab' in channels_list: count += 3
        if 'mask' in channels_list: count += 1
        return count
    Config.INPUT_CHANNELS = _calculate_input_channels([c for c in Config.CHANNELS_TO_USE if c != 'mask']) * 2

def ask_ground_truth(pair):
    print(f"Pair: {pair[0]} & {pair[1]}")
    gt = input("Enter ground truth (D for Diabetic / C for Control): ").strip().upper()
    if gt == 'D': return 'Diabetic'
    if gt == 'C': return 'Control'
    print("Invalid input. Defaulting to 'Control'.")
    return 'Control'

def calc_metrics(results):
    # results: list of dicts with 'gt', 'pred', 'prob'
    tp = sum(1 for r in results if r['gt']=='Diabetic' and r['pred']=='Diabetic')
    tn = sum(1 for r in results if r['gt']=='Control' and r['pred']=='Control')
    fp = sum(1 for r in results if r['gt']=='Control' and r['pred']=='Diabetic')
    fn = sum(1 for r in results if r['gt']=='Diabetic' and r['pred']=='Control')
    total = len(results)
    acc = (tp+tn)/total if total else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    print(f"\nConfusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,'Accuracy':acc,'Precision':prec,'Recall':rec,'F1':f1}

def main():
    images_dir = os.path.join('realdata','images')
    pairs = find_image_pairs(images_dir)
    models = load_ensemble_models()
    
    # Create a simple config object with the required attributes from config module
    class SimpleConfig:
        IMAGE_SIZE = 128  # Use 128 to match trained models, not config.IMAGE_SIZE (256)
        CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask']
        DEVICE = Config.DEVICE
        INPUT_CHANNELS = Config.INPUT_CHANNELS
    
    simple_config = SimpleConfig()
    
    results = []
    for left, right in tqdm(pairs, desc="Predicting pairs", unit="pair"):
        left_path = os.path.join(images_dir, left)
        right_path = os.path.join(images_dir, right)
        pred, prob = predict_with_ensemble(models, left_path, right_path, simple_config)
        gt = ask_ground_truth((left, right))
        results.append({'left':left,'right':right,'pred':pred,'prob':prob,'gt':gt})
        print(f"Predicted: {pred} (prob={prob:.3f}), Ground Truth: {gt}")
    calc_metrics(results)
    # Optionally save results
    with open('realdata_pair_predictions.csv','w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['left','right','pred','prob','gt'])
        writer.writeheader()
        writer.writerows(results)
    print("Results saved to realdata_pair_predictions.csv")

if __name__ == "__main__":
    main()
