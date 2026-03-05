import  cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / 'echonet' / 'raw' / 'EchoNet-Dynamic'
OUT = BASE_DIR / 'echonet' / 'processed'


OUT.mkdir(parents=True, exist_ok=True)

def load_video(path, h=112, w=112):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (w,h)))
    cap.release()
    return np.array(frames, dtype=np.uint8)


def zscore(v):
    v = v.astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-8)


df = pd.read_csv(RAW / 'FileList.csv')
records = []


for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = f"{str(row['FileName']).strip()}.avi"
    vid_path = RAW / 'Videos' / filename
    
    if not vid_path.exists():
        vid_path = RAW / 'Videos' / filename.replace('.AVI', '.avi')

    if not vid_path.exists():
        continue 
    
    video_data = load_video(vid_path)
    
    if video_data.size == 0: continue
    video=zscore(video_data)
    out = OUT / (Path(row['FileName']).stem + '.npy')
    np.save(out, video)
    records.append({'patient_id': Path(row['FileName']).stem, 
                    'ef': row['EF'], 'esv': row['ESV'],
                    'edv': row['EDV'], 'split': row['Split'],
                    'n_frames': video.shape[0], 'out_path': f"echonet/processed/{Path(row['FileName']).stem}.npy"})
    
pd.DataFrame(records).to_csv(OUT / 'labels.csv', index=False)
print(f'Done: {len(records)} videos')
