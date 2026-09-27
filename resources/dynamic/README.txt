Place one dynamic sequence in its own folder, for example:

resources/dynamic/bouncing/frame_000.obj
resources/dynamic/bouncing/frame_001.obj
resources/dynamic/bouncing/frame_002.obj
...

All frames must have the same vertex count, the same triangle connectivity,
and the same vertex ordering/correspondence for the basic Question 8.

Run it with:
python project.py --dynamic-folder resources/dynamic/bouncing
