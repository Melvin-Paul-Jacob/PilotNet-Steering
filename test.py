import cv2
from tensorflow.keras.models import load_model

model = load_model("results/model2/model2.h5", compile=False)
img = cv2.imread('steering_wheel_image.jpg')
#img= cv2.resize(img, (0,0), fx=0.2, fy=0.2) 
rows,cols,_ = img.shape
smoothed_angle = 0
cap = cv2.VideoCapture("video_dataset/data.avi")

x_dim = 64
y_dim = 192

while(1):
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    image= cv2.resize(image,(454,256))
    imagea = image[128-20:-20,35:-35]
    imagea= cv2.resize(imagea,(192,64))
    imagea = cv2.rotate(imagea, cv2.ROTATE_90_CLOCKWISE)
    imagea=imagea/255.0
    imagea= imagea.reshape(1, y_dim, x_dim, 1)
    pred = model.predict(imagea)
    #print(pred[0][0])
    degrees = pred[0][0]
    cv2.imshow('frame', frame)
    # #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    # #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    print("Predicted steering angle: " + str(smoothed_angle) + " degrees")
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        print("-> Ending Video Stream")
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()