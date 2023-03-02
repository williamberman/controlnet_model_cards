# Controlnet

Controlnet is an auxiliary model which augments pre-trained diffusion model with additional conditioning.

Controlnet comes with multiple auxiliary models, each which allows a different type of conditioning

The auxiliary conditioning is passed directly to the diffusers pipeline. If you want to process on image to create the auxiliary conditioning, external dependencies are required.

Controlnet's auxiliary models are trained with stable diffusion 1.5. Experimentally, the auxiliary models can be used with other diffusion models such as dreamboothed stable diffusion.

Some of the additional conditionings can be extracted from images via additional models. We extracted these
additional models from the original controlnet repo into a separate package that can be found on [github](https://github.com/patrickvonplaten/human_pose.git).

## Canny edge detection

Install opencv

```sh
$ pip install opencv-contrib-python
```

```python
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import numpy as np

image = Image.open('images/bird.png')
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-canny",
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
)
pipe.to('cuda')

image = pipe("bird", image).images[0]

image.save('images/bird_canny_out.png')
```

![bird](./images/bird.png)

![bird_canny](./images/bird_canny.png)

![bird_canny_out](./images/bird_canny_out.png)

## M-LSD Straight line detection

Install the additional controlnet models package.

```
$ pip install git+https://github.com/patrickvonplaten/human_pose.git
```

```py
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from human_pose import MLSDdetector

mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

image = Image.open('images/room.png')

image = mlsd(image)

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-mlsd",
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
)
pipe.to('cuda')

image = pipe("room", image).images[0]

image.save('images/room_mlsd_out.png')
```

## Pose estimation

TODO

## Semantic Segmentation

TODO

## Depth control

TODO

## Normal map

TODO

## Scribble

TODO

## HED Boundary

TODO