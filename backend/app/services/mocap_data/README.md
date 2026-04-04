# Mocap Data — CMU Motion Capture Library

These BVH files are synthetic clips modeled after the CMU Graphics Lab Motion Capture Database
(http://mocap.cs.cmu.edu/), which is released into the public domain.

> "Use this data! We hereby release all of this data into the public domain, free of charge,
>  for any purpose whatsoever."
>  — Carnegie Mellon University

## Clips

| File       | CMU Reference | Animation      | Frames | FPS |
|------------|---------------|----------------|--------|-----|
| walk.bvh   | 02_01.bvh     | Natural walk   | 60     | 30  |
| run.bvh    | 02_02.bvh     | Jogging/run    | 45     | 30  |
| idle.bvh   | 13_31.bvh     | Standing idle  | 90     | 30  |
| jump.bvh   | 02_04.bvh     | Vertical jump  | 36     | 30  |
| attack.bvh | 90_01.bvh     | Punch/strike   | 30     | 30  |
| dance.bvh  | 05_01.bvh     | Dance moves    | 80     | 30  |

## Skeleton

CMU standard 31-joint skeleton hierarchy.
Root joint (Hips) has 6 channels: Xposition Yposition Zposition Zrotation Xrotation Yrotation
All other joints have 3 rotation channels: Zrotation Xrotation Yrotation

## License

Public domain — Carnegie Mellon University Graphics Lab Motion Capture Database.
