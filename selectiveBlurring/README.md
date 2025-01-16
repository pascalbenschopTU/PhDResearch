# Privacy preserving blurring

Navigate to selectiveBlurring

```
cd selectiveBlurring
```

## To create one or more descriptors:

The descriptors will be saved as name_1, name_2, name_3...
A matplotlib window will pop up, where you can click to select and save descriptors. 
Once richt clicked (or the window closed), the descriptors are saved.

```
python save_descriptor.py -i <image_path> -n <name>
```


## Applying privacy preserving noise

For images, you can provide one or more descriptor names, e.g. for the descriptors face_1, face_2 etc you can use `-ds face`, which you can combine with other descriptor names.
For device, use `cuda` with nvidia and `mps` with macbook m1+.
```
python video_blur.py -f <video_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> ... -dev <device>
```

```
python video_blur.py -i <image_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> ... -dev <device>
```

Apply directly to videos
```
python video_blur.py -v <video_path> -df output/descriptors/ -ds <descriptor 1> <descriptor 2> ... -dev <device>
```


### MPS device error

If you get this error:
```
NotImplementedError: The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.

```

Then just do this before executing the script
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

