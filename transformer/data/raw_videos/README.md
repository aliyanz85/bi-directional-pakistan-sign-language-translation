# Place your raw video files here

## Directory Structure

Organize your videos in the following structure:

```
raw_videos/
├── word1/
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── video3.mp4
│   ├── video4.mp4
│   ├── video5.mp4
│   ├── video6.mp4
│   ├── video7.mp4
│   └── video8.mp4
├── word2/
│   ├── video1.mp4
│   └── ... (8 videos)
├── word3/
│   └── ... (8 videos)
...
└── word32/
    └── ... (8 videos)
```

## Important Notes

1. **Folder names** will be used as class labels (word names)
2. Each word should have exactly **8 videos** (or adjust in config)
3. Supported formats: `.mp4`, `.avi`, `.mov`
4. Recommended:
   - 30 FPS
   - Good lighting
   - Clear hand visibility
   - Consistent background
   - 3-5 seconds per video

## Video Recording Tips

- Record against a plain background
- Ensure hands are fully visible
- Maintain consistent distance from camera
- Perform the sign clearly and completely
- Include natural variations in speed and position
- Record from different angles for diversity

## After Adding Videos

Run the landmark extraction script:

```bash
python models/training/extract_landmarks.py
```

This will process all videos and create the dataset for training.
