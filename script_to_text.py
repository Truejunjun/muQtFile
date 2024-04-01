from PIL import Image
import os
import cv2
import numpy as np


def defect_position(image):
    height, width = image.shape
    binary = np.where(image > 0, 255, 0).astype('uint8')
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        center = rect[0]
        if center[0] < width/3:
            if center[1] < height/3:
                positions.append("左上")
            elif center[1] < 2*height/3:
                positions.append("正左")
            else:
                positions.append("左下")
        elif center[0] < 2*width/3:
            if center[1] < height/3:
                positions.append("中上")
            elif center[1] < 2*height/3:
                positions.append("中间")
            else:
                positions.append("中下")
        else:
            if center[1] < height/3:
                positions.append("右上")
            elif center[1] < 2*height/3:
                positions.append("正右")
            else:
                positions.append("右下")
    return positions


def detect_defects(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    red_image = np.zeros((height, width), np.uint8)
    blue_image = np.zeros((height, width), np.uint8)

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            if r > 200 and g < 50 and b < 50:  # red
                red_image[y, x] = 255
            elif r < 50 and g < 50 and b > 200:  # blue
                blue_image[y, x] = 255

    # GT的连通域结果
    _, red_labels = cv2.connectedComponents(red_image)
    _, blue_labels = cv2.connectedComponents(blue_image)

    red_count = np.max(red_labels)
    blue_count = np.max(blue_labels)

    red_areas = [np.sum(red_labels == i) for i in range(1, red_count+1)]
    blue_areas = [np.sum(blue_labels == i) for i in range(1, blue_count+1)]

    red_area = max(red_areas) if red_areas else 0
    blue_area = max(blue_areas) if blue_areas else 0

    # 缺陷种类
    sentence = "产品ID为" + os.path.basename(image_path).replace('.png', '') + ". "+ "这个" + image_path.split('/')[-3] + "工艺有"
    if red_count > 0 and blue_count < 3:
        sentence += "有'异物'和'漏固'两个缺陷. "
    elif red_count > 0:
        sentence += "只有'异物'缺陷. "
    elif blue_count < 3:
        sentence += "只有'漏固'缺陷. "

    
    # 异物缺陷
    red_thred = 600
    if red_area > red_thred:
        sentence += "异物较大. "

    if red_count > 5:
        sentence += "异物较多，有" + str(red_count) + "个. "
    elif red_count > 0:
        red_positions = defect_position(red_labels)
        sentence += "异物缺陷位置：{}.".format(red_positions)

    
    # 漏固缺陷
    if blue_count < 2:
        sentence += "漏固严重. "   
    
    blue_positions = defect_position(blue_labels)
    blue_positins_only = ['中上', '中间', '中下']
    blue_positions = [pos for pos in blue_positins_only if pos not in blue_positions]
    if blue_positions:
        sentence += "漏固缺陷位置：{}.".format(blue_positions)

    return sentence


if __name__ == '__main__':
    data_path = '/home/s414f/s414f1/zjy/gemini/Gemini-API/data/miniled_100_test/label'
    images = os.listdir(data_path)
    for image in images:
        print(detect_defects(os.path.join(data_path, image)))
    
