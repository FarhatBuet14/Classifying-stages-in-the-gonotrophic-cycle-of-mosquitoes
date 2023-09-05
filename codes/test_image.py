import numpy as np
np.random.seed(9)
from keras.models import load_model
import cv2
from keras.preprocessing import image
import grad_cam

adult = {0: 'unfed', 1: 'gravid', 2: "semi-gravid", 3: "fully fed"}

model = load_model('../models/EfficientNetB0/model00000533.h5')

image_name = "test_image.jpg"
img = cv2.resize(cv2.imread(image_name), (224, 224))
x = image.img_to_array(img) / 255
res = model.predict(np.array([x]).astype('float32')).tolist()[0]
predicted = adult[res.index(max(res))]
percentage = str(max(res) * 100)

print(image_name + " is " + predicted + " with a probability of " + percentage + "%") 

img = cv2.imread(os.path.join("random_test" , image_name))
height, width, _ = img.shape

superimposed_img = grad_cam.get_cam(model, x, height, width, threshold = 0)

alpha = 0.5
superimposed_img = superimposed_img * alpha + img
cv2.imwrite(f"gradCam_{image_name}", superimposed_img)
