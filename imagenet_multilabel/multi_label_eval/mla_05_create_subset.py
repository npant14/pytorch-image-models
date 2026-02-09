import os
import shutil
import glob

src_dir = "/media/data_cifs/pfeng2/Harmoization/datasets/imagenet_multi_label"
tgt_dir = "/media/data_cifs/pfeng2/Harmoization/datasets/imagenet_mla_subset"
file_paths = glob.glob(os.path.join(src_dir, '*.pth')) 

cnt = 0
for i, src in enumerate(file_paths):
    cnt += 1
    if cnt > 1000:
        break
    print("Successfully copied file %s of %s \r" % (str(i+1), 1000), end="")
    shutil.copy(src, tgt_dir)
print("")
print("Done!")
    

