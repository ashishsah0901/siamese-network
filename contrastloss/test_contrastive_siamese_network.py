from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of testing images")
args = vars(ap.parse_args())

print("[INFO] loading test dataset...")
testImagePaths = list(list_images(args["input"]))
np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(10, 2))
print("[INFO] loading siamese model...")
model = load_model(config.MODEL_PATH, compile=False)

for (i, (pathA, pathB)) in enumerate(pairs):
	imageA = cv2.imread(pathA, 0)
	imageB = cv2.imread(pathB, 0)
	origA = imageA.copy()
	origB = imageB.copy()
	imageB = np.expand_dims(imageB, axis=-1)
	imageA = np.expand_dims(imageA, axis=0)
	imageB = np.expand_dims(imageB, axis=0)
	imageA = imageA / 255.0
	imageB = imageB / 255.0
	preds = model.predict([imageA, imageB])
	proba = preds[0][0]
	fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
	plt.suptitle("Distance: {:.2f}".format(proba))
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(origA, cmap=plt.cm.gray)
	plt.axis("off")
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(origB, cmap=plt.cm.gray)
	plt.axis("off")
	plt.show()