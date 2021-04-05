# here insert the bgr values which you want to convert to hsv
import cv2
import matplotlib.pyplot as plt
import numpy as np

# https://stackoverflow.com/questions/50980810/how-to-create-a-discrete-rgb-colourmap-with-n-colours-using-numpy


def spec(N):
    t = np.linspace(-510, 510, N)
    return np.round(np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255)).astype(
        np.uint8
    )


rgb_colors = spec(10)
color_limits = []
# https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv/51686953
for color in rgb_colors:
    color = color.reshape(1, 1, 3)
    plt.imshow(color)
    plt.show()
    hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    # print(hsvColor)

    lowerLimit = hsvColor[0][0][0] - 10, 100, 100
    upperLimit = hsvColor[0][0][0] + 10, 255, 255

    color_limits.append([upperLimit, lowerLimit])

for color in [
    np.array([0, 0, 0], dtype=np.uint8),
    np.array([255, 255, 255], dtype=np.uint8),
]:
    color = color.reshape(1, 1, 3)
    plt.imshow(color)
    plt.show()
    # print(color)
    hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    # print(hsvColor)

    lowerLimit = hsvColor[0][0][0] - 10, 100, 100
    upperLimit = hsvColor[0][0][0] + 10, 255, 255

    color_limits.append([upperLimit, lowerLimit])


print(color_limits)
