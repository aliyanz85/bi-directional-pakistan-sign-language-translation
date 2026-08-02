import csv, os, sys
p='logs/enhanced_20251203_013652/training.csv'
if not os.path.exists(p):
    print('MISSING_CSV')
    sys.exit(0)
with open(p,'r',encoding='utf-8') as f:
    reader=csv.DictReader(f)
    rows=list(reader)
if not rows:
    print('NO_ROWS')
    sys.exit(0)
last=rows[-1]
best_idx=None
best_val=-1.0
for i,r in enumerate(rows):
    try:
        va=float(r.get('val_accuracy',0))
    except:
        va=0.0
    if va>best_val:
        best_val=va; best_idx=i
print('LAST_EPOCH_INDEX', len(rows)-1)
print('LAST_TRAIN_ACC', float(last.get('accuracy',0)))
print('LAST_VAL_ACC', float(last.get('val_accuracy',0)))
print('LAST_TRAIN_LOSS', float(last.get('loss',0)))
print('LAST_VAL_LOSS', float(last.get('val_loss',0)))
print('BEST_VAL_INDEX', best_idx, 'BEST_VAL_ACC', best_val)
if best_idx is not None:
    br=rows[best_idx]
    print('BEST_EPOCH_TRAIN_ACC', float(br.get('accuracy',0)), 'BEST_EPOCH_VAL_LOSS', float(br.get('val_loss',0)))
