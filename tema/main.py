from compresie_completa import *

# 1
X1 = datasets.ascent()
X1_jpeg = compresie_jpeg(X1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Ascent original - imagine grayscale")
plt.imshow(X1, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Comprimat")
plt.imshow(X1_jpeg, cmap=plt.cm.gray)
plt.axis("off")

plt.savefig('img/grayscale_ascent.pdf', format = 'pdf')
plt.show()

# 2
X2 = datasets.face()
X2_jpeg = compresie_jpeg(X2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Face original - imagine color")
plt.imshow(X2)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Comprimat")
plt.imshow(X2_jpeg)
plt.axis("off")

plt.savefig('img/color_face.pdf', format = 'pdf')
plt.show()

# 3
X3_jpeg = compresie_jpeg(X2, 2700)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Face original - imagine color")
plt.imshow(X2)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Comprimat pana la un mse dat")
plt.imshow(X3_jpeg)
plt.axis("off")

plt.savefig('img/color_face_mse.pdf', format = 'pdf')
plt.show()

X4_jpeg = compresie_jpeg(X1, 150)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Ascent original - imagine grayscale")
plt.imshow(X1, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Comprimat pana la un mse dat")
plt.imshow(X4_jpeg, cmap=plt.cm.gray)
plt.axis("off")

plt.savefig('img/ascent_grayscale_mse.pdf', format = 'pdf')
plt.show()

# 4
video_path = "video.mp4"
cap = cv.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frames.append(frame_rgb)
cap.release()
X_vid = np.array(frames)
X_vid_jpeg = compresie_jpeg(X_vid)
num_frames = X_vid_jpeg.shape[0]
h, w = X_vid_jpeg.shape[1:3]
fps = 30
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('img/video_reconstruit.mp4', fourcc, fps, (w, h))