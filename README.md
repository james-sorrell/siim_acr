# SIIM-ACR Pneumothorax Segmentation

> Identify Pneumothorax disease in chest x-rays

Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening event.

Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be 
useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

> Kaggle Page

All data and information regarding this project has been taken from the following Kaggle Competition
  https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/description

> Sandbox | Ideas and Notes

- Some of the training images seem poor in quality (See imgs/11.png), I have noticed multiple training images that look like this. It could be beneficial to remove this data from the training. One of the images was upside down (See prediction_examples/6.png).

- The images seem to be at varying scales and rotations, this could make the problem more complicated than it potentially needs to be. This can potentially be adressed by isolating the chest/lung area from the image/mask before scaling.

- Potentially create a separate Model for inference on Men and Women. Images are significantly different, could be room for potential prediction improvements.