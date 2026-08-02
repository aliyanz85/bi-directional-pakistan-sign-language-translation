import time
import numpy as np
import os
from tensorflow import keras

MODEL_PATH = 'models/sign_language_model_final.h5'
DATA_PATH = 'processed_data/dataset.npz'
REPEAT = 50  # how many times to repeat prediction for averaging

if not os.path.exists(MODEL_PATH):
    print('MISSING_MODEL')
    raise SystemExit(1)
if not os.path.exists(DATA_PATH):
    print('MISSING_DATASET')
    raise SystemExit(1)

# Load model
model = keras.models.load_model(MODEL_PATH)
# Load dataset (use X_test if present)
data = np.load(DATA_PATH)
if 'X_test' in data:
    X_test = data['X_test']
else:
    X_test = data['X']

# Use up to 10 sequences for benchmark
n_seq = min(10, len(X_test))
sequences = X_test[:n_seq]

# Warm up
print('Warmup prediction...')
_ = model.predict(np.expand_dims(sequences[0], 0))

# measure per-sequence prediction time (model.predict single sequence)
timings = []
for seq in sequences:
    seq_in = np.expand_dims(seq, 0)
    # repeat predictions to get stable timing
    t0 = time.time()
    for _ in range(REPEAT):
        _ = model.predict(seq_in, verbose=0)
    t1 = time.time()
    avg = (t1 - t0) / REPEAT
    timings.append(avg)
    print(f'Sequence avg latency: {avg*1000:.3f} ms')

overall_avg = np.mean(timings)
overall_std = np.std(timings)
print(f'Overall avg per-sequence latency: {overall_avg*1000:.3f} ms (+/- {overall_std*1000:.3f} ms)')

# Save results
out = {
    'n_sequences': int(n_seq),
    'repeat': int(REPEAT),
    'timings_seconds': [float(t) for t in timings],
    'avg_seconds': float(overall_avg),
    'std_seconds': float(overall_std)
}
import json
with open('logs/evaluation/inference_benchmark.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved logs/evaluation/inference_benchmark.json')
