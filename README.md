# Privacy preserving blurring

Navigate to selectiveBlurring

```
cd selectiveBlurring
```

Then do the following command:

```

python video_blur.py -f <video_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> -dev <device>

python video_blur.py -f ../Action-Recognition/data/UCF101-frames/UCF101/v_Haircut_g23_c04 -df output/descriptors/ -ds face -dev mps

```

```
python video_blur.py -i <image_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> -dev <device>

python video_blur.py -i ../extra/Street_sample.png -df output/descriptors/ -ds face -dev mps
```

To create one or more descriptors:

The descriptors will be saved as name_1, name_2, name_3...

```
python save_descriptor.py -i <image_path> -n <name>

python save_descriptor.py -i ../extra/Street_sample.png -n arm
```



