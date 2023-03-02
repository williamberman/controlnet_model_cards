# Controlnet

Controlnet is an auxiliary model which augments pre-trained diffusion model with additional conditioning.

Controlnet comes with multiple auxiliary models, each which allows a different type of conditioning

The auxiliary conditioning is passed directly to the diffusers pipeline. If you want to process on image to create the auxiliary conditioning, external dependencies are required.

Controlnet's auxiliary models are trained with stable diffusion 1.5. Experimentally, the auxiliary models can be used with other diffusion models such as dreamboothed stable diffusion.

## Canny edge detection

Canny edge detection has a dependency on opencv

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

TODO

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
