import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split


SEED = 42

#EchoNet -- use provided splits
df = pd.read_csv('data/echonet/processed/labels.csv')
for sp in ['TRAIN', 'VAL', 'TEST']:
    df[df.split==sp].to_csv(f'data/echonet/splits/{sp.lower()}.csv', index=False)


#CAMUS -- stratified by EF < 45%
df2 = pd.read_csv('data/camus/processed/labels.csv')
pts = df2[df2.view=='4CH'][['patient_id', 'ef']].drop_duplicates()
pts['group'] = (pts.ef < 45).astype(int)
train, test = train_test_split(pts, test_size=0.10, stratify=pts.group, random_state=SEED)
train, val = train_test_split(train, test_size=0.11, stratify=train.group, random_state=SEED)
for name, subset in [('train', train), ('val', val), ('test', test)]:
    ids = subset.patient_id.tolist()
    df2[df2.patient_id.isin(ids)].to_csv(f'data/camus/splits/{name}.csv', index=False)
    print(f'CAMUS {name}: {len(ids)} patients')

