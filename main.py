import cv2
import os
import csv
import datetime
import numpy as np
import mediapipe as mp
from dedectPose import detectPose
from classifypose import classifyPose
from calculateangel import calculateAngle
import time

def save_frame_as_png(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    csv_filename = "frames_info.csv"
    csv_path = os.path.join(output_folder, csv_filename)

    # mp_pose = mp.solutions.pose
    # with mp_pose.Pose(upper_body_only=False) as pose_tracker:
    #     result = pose_tracker.process(image=frame)
    #     pose_landmarks = result.pose_landmarks
    #     print(pose_landmarks)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Time", "File Name", "Action Type"])
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                # Burada model uygulamasının fonksiyonu çağırılacak
                # frame = cv2.flip(frame, 1)
    
                # Get the width and height of the frame
                frame_height, frame_width, _ =  frame.shape
                
                # Resize the frame while keeping the aspect ratio.
                frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
                
                # Perform Pose landmark detection.
                mp_pose = mp.solutions.pose
                pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
                frame, landmarks = detectPose(frame, pose_video, display=False)            
        
                # Check if the landmarks are detected.
                if landmarks:                                       
                    # Perform the Pose Classification.
                    frame, _ = classifyPose(landmarks, frame, display=False)
                    output_image, label = classifyPose(landmarks, frame, display=False)

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
                    

                    new_data = np.array([[left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,left_knee_angle, right_knee_angle]])
                    predictions = new_model.predict(new_data)
                    predicted_class = np.argmax(predictions, axis=1)


                    if predicted_class == 0:
                        label = "idle"
                        frame_name = f"frame_{frame_count:04d}.png"
                        frame_path = os.path.join(output_folder, frame_name)
                        # cv2.imwrite(frame_path, frame)   Bu durumda ekran görüntüsü almıyor.
                        seconds = round(frame_count / fps)
                        milliseconds = round((seconds - int(seconds)) * 1000)
                        time_format = str(datetime.timedelta(seconds=int(seconds), milliseconds=milliseconds))
                        csv_writer.writerow([time_format, frame_name, label])


                    elif predicted_class == 1:
                        label = "shot"
                        frame_name = f"frame_{frame_count:04d}.png"
                        frame_path = os.path.join(output_folder, frame_name)
                        cv2.imwrite(frame_path, frame)
                        seconds = round(frame_count / fps)
                        milliseconds = round((seconds - int(seconds)) * 1000)
                        time_format = str(datetime.timedelta(seconds=int(seconds), milliseconds=milliseconds))
                        csv_writer.writerow([time_format, frame_name, label])
                        print(time_format, frame_name, "ekran görüntüsü kaydedildi")     

                    else:
                        label = "unknown"  

                    frame_count += 1
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("frame", frame)


                else:
                    print("landmark'lar alınamadı")
                    frame_count += 1

            except Exception as e:
                print(f"Hata oluştu: {str(e)}")
                continue
         

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break                  

        cap.release()
        cv2.destroyAllWindows()

    print(f"Frame'ler başarıyla kaydedildi: {output_folder}")
    print(f"CSV dosyası oluşturuldu: {csv_path}")

if __name__ == "__main__":
    video_path = "datacollection/media/video720p.mp4"
    output_folder = "results"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    import tensorflow as tf
    new_model = tf.keras.models.load_model('datacollection/my_model')
    print("tensorflow modeli yüklendi")
    
    save_frame_as_png(video_path, output_folder)
