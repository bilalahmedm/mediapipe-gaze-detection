import cv2
import mediapipe as mp
import numpy as np
import math

face_input = 'media/face.mp4'
front_input = 'media/front.mp4'
face_output = 'output/face_outupt.mp4'
front_output = 'output/front_output.mp4'
fps = 24
size = 0.5                            #between 0 and 1
distance_from_center = 0              # +ve distance means below center, -ve distance means above center
x_size = 100
x_border = 200
brightness = 0

x1,y1,x2,y2 = 0,0,0,0
iris_pos = ''
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE= [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]  
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

cv2.namedWindow('Face Camera', cv2.WINDOW_NORMAL)
cv2.namedWindow('Front Camera', cv2.WINDOW_NORMAL)
face_camera = cv2.VideoCapture(face_input)
front_camera = cv2.VideoCapture(front_input)
if face_input == 0:
    face_camera.set(3,1280)
    face_camera.set(4,720)  
w1 = int(face_camera.get(3))
h1 = int(face_camera.get(4))  
w = int(front_camera.get(3))
h = int(front_camera.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(face_output,fourcc,fps,(w1,h1))
out2 = cv2.VideoWriter(front_output,fourcc,fps,(w,h))

center_y = h/2 + distance_from_center
ysize = h/2 * size
top_point_y = int(center_y - ysize)
bottom_point_y = int(center_y + ysize)


def fillPolyTrans(img, points, color, opacity):
    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = img.copy()  # coping the image
    cv2.fillPoly(overlay,[list_to_np_array], color )
    new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    # print(points_list)
    img = new_img
    cv2.polylines(img, [list_to_np_array], True, color,1, cv2.LINE_AA)
    return img

def euclidean_distance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def iris_position(iris_center, left_point, right_point):
    total_distance = euclidean_distance(left_point, right_point)
    center_to_right_distance = euclidean_distance(iris_center, right_point)
    ratio = center_to_right_distance / total_distance
    if ratio <= 0.4:
        iris_pos = 'RIGHT'
    elif ratio > 0.4 and ratio <= 0.6:
        iris_pos = 'CENTER'
    else:
        iris_pos = 'LEFT'
    return iris_pos, ratio

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    
    landmarks_list = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in landmarks_list]
        
    return landmarks_list

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    while face_camera.isOpened():
        ret1, frame1 = face_camera.read()
        ret2, frame2 = front_camera.read()
        if not ret1:
            print('Empty camera frame')
            break
        if not ret2:
            print('Empty fornt frame')
            break
            
        if face_input == 0:
            frame1 = cv2.flip(frame1,1)
        if brightness > 0:
            frame1 = np.int32(frame1) + brightness
            frame1 = np.clip(frame1, 0, 255)
            frame1 = np.uint8(frame1)
        frame1.flags.writeable = False
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame1)
        
        frame1.flags.writeable = True
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            landmarks_list = landmarksDetection(frame1,results,draw=False)
            
            frame1 = fillPolyTrans(frame1, [landmarks_list[p] for p in LEFT_EYE], (0,255,0), opacity=0.3)
            frame1 = fillPolyTrans(frame1, [landmarks_list[p] for p in RIGHT_EYE], (0,255,0), opacity=0.3)
            frame1 = fillPolyTrans(frame1, [landmarks_list[p] for p in LEFT_IRIS], (0,0,255), opacity=0.2)
            frame1 = fillPolyTrans(frame1, [landmarks_list[p] for p in RIGHT_IRIS], (0,0,255), opacity=0.2)
            
            l_cx = int((landmarks_list[LEFT_IRIS[0]][0] + landmarks_list[LEFT_IRIS[2]][0])/2)
            l_cy = int((landmarks_list[LEFT_IRIS[1]][1] + landmarks_list[LEFT_IRIS[3]][1])/2)
            r_cx = int((landmarks_list[RIGHT_IRIS[0]][0] + landmarks_list[RIGHT_IRIS[2]][0])/2)
            r_cy = int((landmarks_list[RIGHT_IRIS[1]][1] + landmarks_list[RIGHT_IRIS[3]][1])/2)
            
            
            cv2.circle(frame1, (l_cx, l_cy), 4, (0,0,255), -1)
            cv2.circle(frame1, (r_cx, r_cy), 4, (0,0,255), -1)
            
            left_position, left_ratio = iris_position((l_cx,l_cy), landmarks_list[L_H_LEFT[0]], landmarks_list[L_H_RIGHT[0]])
            right_position, right_ratio = iris_position((r_cx,r_cy), landmarks_list[R_H_LEFT[0]], landmarks_list[R_H_RIGHT[0]])
            
            ratio = (left_ratio + right_ratio)/2
            ratio=round(ratio,3)
            
            if ratio <= 0.43:
                iris_pos = 'RIGHT'
                x1 = int(x_border)
                y1 = top_point_y
                x2 = int(w/3 - x_size)
                y2 = bottom_point_y
            elif ratio > 0.43 and ratio <= 0.57:
                iris_pos = 'CENTER'
                x1 = int(w/3 + x_size)
                y1 = top_point_y
                x2 = int(2*w/3 - x_size)
                y2 = bottom_point_y
            else:
                iris_pos = 'LEFT'
                x1 = int(2*w/3 + x_size)
                y1 = top_point_y
                x2 = int(w - x_border)
                y2 = bottom_point_y
                
            cv2.rectangle(frame2,(x1,y1),(x2,y2),(0,0,255),5)     

        out1.write(frame1)
        out2.write(frame2)
        cv2.imshow('Face Camera', frame1)
        cv2.imshow('Front Camera', frame2)
        k = cv2.waitKey(1)
        if k == 27:
            break
out1.release()
out2.release()
face_camera.release()
front_camera.release()
cv2.destroyAllWindows()