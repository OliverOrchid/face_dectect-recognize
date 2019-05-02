"""
采集人脸图像
"""
import cv2
def generate():

  camera = cv2.VideoCapture(0)#打开摄像头，参数0表示调用默认摄像头
  count = 0#计数器
  delta=0#timer
  while (True):


    ret, frame = camera.read()#返回两值，后者将被调用
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#灰度化处理

    if delta%3==0:
      cv2.imwrite('./dataset/temp/%s.jpg' % str(count), frame)#存放至指定目录
      print (count)
      count += 1
    
    delta += 1

    cv2.imshow("人脸图像采集中>>>", frame)
    if cv2.waitKey(42) & 0xff == ord("q"):#电影帧数一般为24FPS
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  generate()
