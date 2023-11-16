import cv2
import numpy
import os
import mediapipe as mp
from dedectPose import detectPose
from classifypose import classifyPose
from calculateangel import calculateAngle
import csv



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

def imagedatacollection(folder_path, output_folder):

    csv_filename = "frames_info.csv"
    csv_path = os.path.join(output_folder, csv_filename)


    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Action Type", "left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle", "left_knee_angle", "right_knee_angle"])



        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):        
                file_path = os.path.join(folder_path, filename)

                image = folder_path + filename


                try:
                    image = cv2.imread(image)
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    image_hight, image_width, _ = image.shape
                    if not results.pose_landmarks:
                        print("landmark bulunamadı")
                        continue
                    # print(
                    #     f'Nose coordinates: ('
                    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
                    # )

                    # Draw pose landmarks.
                    # print(f'Pose landmarks of {filename}:')
                    annotated_image = image.copy()
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

                    image, landmarks = detectPose(image, pose, display=False) 
                        
                    # Perform the Pose Classification.
                    image, _ = classifyPose(landmarks, image, display=False)
                    output_image, label = classifyPose(landmarks, image, display=False)


                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]



                    left_elbow_angle = round(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]))

                    # Get the angle between the right shoulder, elbow and wrist points. 
                    right_elbow_angle = round(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]))   

                    # Get the angle between the left elbow, shoulder and hip points. 
                    left_shoulder_angle = round(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]))

                    # Get the angle between the right hip, shoulder and elbow points. 
                    right_shoulder_angle = round(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]))

                    # Get the angle between the left hip, knee and ankle points. 
                    left_knee_angle = round(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))

                    # Get the angle between the right hip, knee and ankle points 
                    right_knee_angle = round(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))




                    csv_writer.writerow(["idle", left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,left_knee_angle, right_knee_angle])
                    print("kaydedildi")
                    
                    cv2.waitKey(0) 
                    # Close all windows
                    cv2.destroyAllWindows()

                except Exception as e:
                    print(f"Hata oluştu: {filename} - {str(e)}")




if __name__ == "__main__":
    folder_path = "datacollection/idle/"
    output_folder = "datacollection/idle-csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    imagedatacollection(folder_path, output_folder)

