import numpy as np, os, sys
npz='processed_data/dataset.npz'
if not os.path.exists(npz):
    print('MISSING_DATASET')
    sys.exit(0)
else:
    data=np.load(npz)
    keys=list(data.files)
    print('FILES_IN_NPZ:', keys)
    def sh(k):
        a=data[k]
        print(k, a.shape)
    for k in keys:
        try:
            sh(k)
        except Exception as e:
            print('ERR', k, e)

# sample video properties for first files in good and funny if present
import cv2
for cls in ['good','funny']:
    p=os.path.join('videosDataset', cls)
    if os.path.isdir(p):
        files=[f for f in os.listdir(p) if f.endswith('.mp4')]
        if files:
            path=os.path.join(p,files[0])
            cap=cv2.VideoCapture(path)
            fps=cap.get(cv2.CAP_PROP_FPS)
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps>0 else 0
            print(f'VIDEO_SAMPLE {cls}:', files[0], 'FPS', fps, 'FRAMES', frames, 'DURATION_s', round(duration,2),'W',w,'H',h)
            cap.release()
        else:
            print('NO_VIDEOS_IN',cls)
    else:
        print('NO_CLASS_FOLDER',cls)
