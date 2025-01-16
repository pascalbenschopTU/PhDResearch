# Privacy preserving blurring

Navigate to selectiveBlurring

```
cd selectiveBlurring
```


For images, you can provide one or more descriptor names. For face_1, face_2 etc you can use `-ds face`
```
python video_blur.py -f <video_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> ... -dev <device>
```

```
python video_blur.py -i <image_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> ... -dev <device>
```

Apply directly to videos
```
python video_blur.py -v <video_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> ... -dev cuda
```

## To create one or more descriptors:

The descriptors will be saved as name_1, name_2, name_3...

```
python save_descriptor.py -i <image_path> -n <name>
```


