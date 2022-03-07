import cv2
import mediapipe as mp

import numpy as np

frame_count = 0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

crop_flag = False
draw = False

folder_path = '/Users/anuj07/Desktop/UCLA_Q2/239/Project_Videos'
video_path = '/set1/video_front.mp4'
cap = cv2.VideoCapture(folder_path + video_path)


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
    #skin classfication
    # Get pointer to video frames from primary device
    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
    image = skinYCrCb

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        ptc = face_landmarks.landmark[4]
        if not crop_flag:
          pt1 = face_landmarks.landmark[10]
          pt2 = face_landmarks.landmark[152]
          pt3 = face_landmarks.landmark[234]
          pt4 = face_landmarks.landmark[454]
          ymin = int(pt1.y*1080) - 50
          ymax = int(pt2.y*1080) + 50
          xmin = int(pt3.x*1920) - 50
          xmax = int(pt4.x*1920) + 50
          y_dif = (ymax - ymin)//2
          x_dif = (xmax - xmin)//2
          # crop_flag = True

        if draw == True:
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())


    image = image[int(pt1.y*1080)-20:int(pt2.y*1080)+100, int(pt3.x*1920)-20:int(pt4.x*1920)+20]

    r = np.mean(image[:,:,0])
    g = np.mean(image[:,:,1])
    b = np.mean(image[:,:,2])
    # print(r,g,b)
    if frame_count  == 0:
      mean_rgb = np.array([r,g,b])
    else:
      mean_rgb = np.vstack((mean_rgb,np.array([r,g,b])))

    frame_count += 1

    cv2.imshow('Image', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
  np.save("rgb.npy", mean_rgb)
cap.release()
