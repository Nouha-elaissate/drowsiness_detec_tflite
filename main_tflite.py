import cv2
import mediapipe as mp
import math
import time
from collections import deque                           #deque fil d'attente collect: biblio
import psycopg2
import sys


#identifiants de connex
DB_HOST = "serveur-somnolence-nouha-final.postgres.database.azure.com"
DB_USER = "nadmin"
DB_PASSWORD = "Blue2025@"
DB_NAME = "postgres"
DB_PORT = "5432"
DB_SSL_MODE = "require"

RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]          #indice des repères spécifique de REYE
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]          #indice LEYE

MOUTH_LANDMARKS = [78, 82, 13, 312, 308, 317, 14, 87]          #idx bouche

EAR_THRESHOLD = 0.2                                     #seuil EAR
CLOSED_THRESH = 60

MAR_THRESHOLD = 0.5                                     #seuil MAR
YAWN_COUNT_LIMIT = 3
YAWN_TIME_WINDOW = 300
MOUTH_OPEN_FRAMES = 60

FPS = 30                                                #seuil PERCLOS
PERCLOS_TIME_WINDOW_SECONDS = 60
PERCLOS_THRESHOLD = 80

def get_pix_coords(landmarks, indices, img_h, img_w) :          #éviter la repetion du m^ code REYE LEYE cette boucle calcule
   points = []                                                  #les coords non normalisé                                                    
   for idx in indices:
      lm = landmarks[idx]
      points.append((int(lm.x*img_w) , int(lm.y*img_h)))
   return points 

def calculate_distance(p1,p2):
   return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)            #calcul dist euc entre p1 et p2 (des tuples x,y)

def eye_aspect_ratio(eye):              
    A = calculate_distance(eye[1],eye[5])                           #calcul des distances verticale A et B et horiz C
    B = calculate_distance(eye[2],eye[4])                           #les indices de eye sont l'ordre des pts de repères
    C = calculate_distance(eye[0],eye[3])                           #33:0, 160:1, 158:2, 133:3, 153:4, 144:5
    if C ==0.0:
       return 0.0
    
    ear = (A+B) / (2*C)
    return ear

def mouth_aspect_ratio(mouth):
   A = calculate_distance(mouth[1],mouth[7])                   #A,B,C dist vertcl et D horiz
   B = calculate_distance(mouth[2],mouth[6])
   C = calculate_distance(mouth[3],mouth[5])
   D = calculate_distance(mouth[0],mouth[4])
   if D == 0.0:
      return 0.0
   mar = (A+B+C) / (3*D)
   return mar

def enregistrer_alerte_bd(driver_id, alert_name, metric_value, details=""):
   connection = None
   alert_type_id = None
   try:
    print("Tentative de connexion à la base de données Azure")
    connection = psycopg2.connect(         
        host = DB_HOST,
        database = DB_NAME,
        user = DB_USER,
        password = DB_PASSWORD,
        port = DB_PORT,
        sslmode = DB_SSL_MODE
    )
    cursor = connection.cursor()

    sql_select_query = "SELECT alert_type_id FROM alert_types WHERE alert_name = %s"
    cursor.execute(sql_select_query, (alert_name,))
    result = cursor.fetchone()
    if result is None:
       print(f"--- ERREUR : Le type d'alerte '{alert_name}' n'a pas été trouvé dans la BD.")
       return
    alert_type_id = result [0]


    sql_insert_query = "INSERT INTO drowsiness_events (driver_id, alert_type_id, metric_value, details) VALUES (%s, %s, %s, %s);"
    cursor.execute(sql_insert_query,(driver_id, alert_type_id, metric_value, details))
    connection.commit()
    print(f"Événement '{alert_name}' enregistré avec succès pour le conducteur ID {driver_id}.")
       
   except Exception as error:
      print(f"---ERREUR BD: {error}")
   finally:
        if connection:
            cursor.close()
            connection.close()

def flux_video():
    #Initalisation de la capture de vidéo
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():                          #Vérifier que la caméra fonctionne 
        print("Impossible d'ouvrir la caméra")
        return
    
    print("Caméra initialisée avec succés. Appuyer sur 'q' pour quitter")
    
    mp_face_mesh = mp.solutions.face_mesh                       #initialiser mediapipe facemesh 
    mp_drawing = mp.solutions.drawing_utils                     # dessiner les pts de repère & connexions
    mp_drawing_styles = mp.solutions.drawing_styles             #style de dessin

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,             #config de detection (flux vidéo non image)
                                       max_num_faces=1,                         #detection 1 visage conducteur
                                       refine_landmarks=True,                      #précision autour des yeux
                                       min_detection_confidence=0.5,            #seuil minim de surete de detction et du meme visage
                                       min_tracking_confidence=0.5)
    
    closure_counter = 0                                               #initial compteur EAR

    yawn_counter = 0                                                    #initial compteurs MAR + var
    yawn_start_time = None                                              
    is_yawning = False
    mouth_open_counter = 0

    perclos_alert_sent = False
    max_frames = PERCLOS_TIME_WINDOW_SECONDS * FPS                      #inital var PERCLOS
    eye_status_history = deque(maxlen=max_frames)                       #fil d'attente dont la capacité max est max_frames
   

    #Boucle capture et addichage de frames en continu
   
    while True:
        ret,frame = cap.read()                      #Lecture de frame
       
        if not ret :
            print("Impossible de lire la frame.")                   #Vérification de la lecture de frame
            break
        
        image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)           #convertir la frame en RGB
        results = face_mesh.process(image_rgb)                      #traitement image avec mediapipe facemesh
        annotated_image = frame.copy()

        ear = 0.0   
        mar = 0.0
        perclos = 0.0

        if results.multi_face_landmarks:                        
           for face_landmarks in results.multi_face_landmarks:
              img_h, img_w, _ = annotated_image.shape                      #hauteur et largeur de l'image
              
              right_eye_points = get_pix_coords(face_landmarks.landmark, RIGHT_EYE_LANDMARKS, img_h, img_w)         #conversion pix
              left_eye_points = get_pix_coords(face_landmarks.landmark, LEFT_EYE_LANDMARKS, img_h, img_w)
              mouth_points = get_pix_coords(face_landmarks.landmark, MOUTH_LANDMARKS, img_h, img_w)
              
              if len(right_eye_points) == 6 and len(left_eye_points) == 6:
               ear_right = eye_aspect_ratio(right_eye_points)
               ear_left = eye_aspect_ratio(left_eye_points)
               ear = (ear_right + ear_left) / 2.0     

              if len(mouth_points) == 8:
                 mar = mouth_aspect_ratio(mouth_points)

            #boucle de detection micro-sommeil

              if ear < EAR_THRESHOLD:
                 closure_counter += 1
                 eye_status_history.append(1)
              else:
                 closure_counter = 0
                 eye_status_history.append(0)

            #boucle de detection de baillement 

              if mar > MAR_THRESHOLD:
                 mouth_open_counter += 1
              else:
                 mouth_open_counter = 0
                 if is_yawning :
                    is_yawning = False

            #confirmed yawn

              if mouth_open_counter >= MOUTH_OPEN_FRAMES and not is_yawning:
                 is_yawning = True
                 yawn_counter += 1

                 if yawn_counter == 1:
                    yawn_start_time = time.time()
            
              mp_drawing.draw_landmarks(
                image=annotated_image,
                 landmark_list=face_landmarks,
                 connections=mp_face_mesh.FACEMESH_CONTOURS,
                 landmark_drawing_spec=None,
                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        else:
           closure_counter = 0
           eye_status_history.append(0)                                 #si aucun vis n'est detec

        #logique d'alerte avec priorité

        alert_text = ""
        DRIVER_ID_TEST = 1

        if len(eye_status_history) == max_frames:                      
            closed_frames = sum(eye_status_history)
            perclos = (closed_frames / max_frames) * 100
        

        if closure_counter >= CLOSED_THRESH:
           alert_text = "ALERT: MICROSLEEP"
           if closure_counter == CLOSED_THRESH:
              enregistrer_alerte_bd(DRIVER_ID_TEST, "microsleep", ear)
        elif yawn_counter >= YAWN_COUNT_LIMIT:
           if yawn_start_time is not None and time.time() - yawn_start_time < YAWN_TIME_WINDOW:
                alert_text = "ALERT: YAWN DROWSINESS"
                if yawn_counter == YAWN_COUNT_LIMIT:
                  enregistrer_alerte_bd(DRIVER_ID_TEST, "yawn frequency", yawn_counter)
                  yawn_counter = 0
                  yawn_start_time = None
        elif perclos >= PERCLOS_THRESHOLD:
           alert_text = "ALERT: FATIGUE"  
           if not perclos_alert_sent:
              enregistrer_alerte_bd(DRIVER_ID_TEST, "fatigue", perclos)  
              perclos_alert_sent = True
        else:
              perclos_alert_sent = False
              
        if yawn_start_time is not None and time.time() - yawn_start_time > YAWN_TIME_WINDOW:
            yawn_counter = 0
            yawn_start_time = None                                                       

        #affichage d'alerte

        if alert_text:
             cv2.putText(annotated_image, alert_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(annotated_image, f"AVG EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    #afficheear
        cv2.putText(annotated_image,f"MAR: {mar:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)         #affichemar
        cv2.putText(annotated_image,f"Yawns: {yawn_counter}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)    #affiche freq de yawn
        cv2.putText(annotated_image, f"PERCLOS: {perclos:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  #afficher perclos

        cv2.imshow('Détection de somnolence', annotated_image)                 #Affichage de la frame dans une fenêtre
        
        key = cv2.waitKey(1)&0xFF                                      #s"assurer si des touches ont été préssées
        if key == ord('q') :                                               #si on touche q pour quitter prog
         print("Arrêt programme")
         break
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()                               #fermer le modèle mediapipe
flux_video()
