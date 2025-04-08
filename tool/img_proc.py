import os
from PIL import Image

source_imgs_dir= "/home0/students/master/2022/wangzy/OR_image/TGRS/A2-FPN/"
target_imgs_dir= "/home0/students/master/2022/wangzy/predict_Vimage/V_TGRS/A2-FPN/"
if not os.path.exists(target_imgs_dir):
    os.mkdir(target_imgs_dir)

for file in os.listdir(source_imgs_dir):
    im = Image.open(source_imgs_dir + file)
    out = im.resize((256, 256), Image.ANTIALIAS)
    out.save(target_imgs_dir + file)


