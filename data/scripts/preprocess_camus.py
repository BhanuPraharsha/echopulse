import SimpleITK as sitk
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

RAW = Path('data/camus/raw/database_nifti')
OUT = Path('data/camus/processed')
OUT.mkdir(parents=True, exist_ok=True)

def load_nii(p):
    img = sitk.ReadImage(str(p))
    return sitk.GetArrayFromImage(img).astype(np.float32)

def align_crop(video, mask, pad=10):
    m = mask[1] if mask.ndim == 3 else mask
    ys, xs = np.where(m > 0)
    if len(ys) == 0: return video
    y1, y2 = max(ys.min()-pad, 0), min(ys.max()+pad, m.shape[0])
    x1, x2 = max(xs.min()-pad, 0), min(xs.max()+pad, m.shape[1])
    cropped = video[:, y1:y2, x1:x2]
    return np.stack([cv2.resize(f, (112, 112)) for f in cropped])

def get_ef(cfg):
    for l in Path(cfg).read_text().splitlines():
        if l.startswith('EF:'): return float(l.split(':')[1].strip())
    return float('nan')

records = []
for pdir in tqdm(sorted([p for p in RAW.glob('patient????') if p.is_dir()])):
    pid = pdir.name
    
    cfg = pdir / 'Info_4CH.cfg'
    ef = get_ef(cfg) if cfg.exists() else float('nan')
    
    for view in ['2CH', '4CH']:
        seq = pdir / f'{pid}_{view}_half_sequence.nii.gz'
        mask = pdir / f'{pid}_{view}_half_sequence_gt.nii.gz'
        
        if not seq.exists():
            print(f"Warning: Missing sequence file {seq}")
            continue
            
        video = load_nii(seq)
        
        if mask.exists(): 
            video = align_crop(video, load_nii(mask))
            
        video = (video - video.mean()) / (video.std() + 1e-8)
        
        out = OUT / f'{pid}_{view}.npy'
        np.save(out, video.astype(np.float32))
        
        records.append({
            'patient_id': pid, 
            'view': view, 
            'ef': ef,
            'n_frames': video.shape[0], 
            'out_path': str(out)
        })

pd.DataFrame(records).to_csv(OUT / 'labels.csv', index=False)
print(f'Done: {len(records)} sequences')
