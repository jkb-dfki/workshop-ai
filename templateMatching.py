import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./slides/image.png',0)
template = cv2.imread('./slides/template.png',0)
template_width, template_height = template.shape[::-1]


response = cv2.matchTemplate(img, template,cv2.TM_CCOEFF)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response)
top_left = max_loc

bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

img_cp = img.copy()
cv2.rectangle(img_cp,top_left, bottom_right, 255, 2)



plt.subplot(231),plt.imshow(img,cmap = 'gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232),plt.imshow(template,cmap = 'gray')
plt.title('Template'), plt.xticks([]), plt.yticks([])

plt.subplot(233),plt.imshow(response,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

plt.subplot(234),plt.imshow(img_cp,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

plt.show()
