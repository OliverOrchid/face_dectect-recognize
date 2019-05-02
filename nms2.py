import cv2
import numpy as np

def nms(bounding_boxes, confidence_score, threshold):
    bboxes = np.array(bounding_boxes)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    score = np.array(confidence_score)#可信度

    picked_boxes = []
    picked_score = []

    areas =(x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(score)#排序

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])

        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def main():

    img_path = "./dataset/test_images/pic-testForNMS.png"
    image = cv2.imread(img_path)
    orig = image.copy()


    #为了便于演示，我们在这手动设置了三个矩形框
    bounding_boxes = [(117, 72, 287, 327), (150, 67, 305, 282),
            (246, 121, 368, 304)]
    confidence_score = [0.9, 0.7, 0.3]


    # 初始化参数
    font = cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
    font_scale = 0.6
    thickness = 1
    threshold = 0.3

    for (x1, y1, x2, y2), confidence in zip(bounding_boxes, confidence_score):
        (w, h), baseline = cv2.getTextSize(
                str(confidence), font, font_scale, thickness)
        cv2.rectangle(orig, (x1, y1 - (2 * baseline + 5)),#颜色，顶点
                (x1 + w, y1), (0, 255, 255), -1)
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 255), 2)

        #添加文字，font_scale表示字体大小
        cv2.putText(orig, str(confidence), (x1, y1),
                font, font_scale, (0, 0, 0, thickness))


    # 绘制图框
    picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)

    for (x1, y1, x2, y2), confidence in zip(picked_boxes, picked_score):
        (w, h), baseline = cv2.getTextSize(
                str(confidence), font, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - (2 * baseline + 5)),
                (x1 + w, y1), (0, 0, 255), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(image, str(confidence), (x1, y1),
                font, font_scale, (0, 0, 0, thickness))

    while True:
        cv2.imshow("Original Pic>>>", orig)
        cv2.imshow("After NMS>>>", image)

        # myKey=cv2.waitKey(0)#ESC即可退出

        if cv2.waitKey(1000) & 0xff==ord('q'):
            print("窗口即将关闭")
            break
    cv2.destroyAllWindows

        # if cv2.waitKey(42) & 0xff == ord("q"):#电影帧数一般为24FPS
if __name__ == "__main__":
    main()
