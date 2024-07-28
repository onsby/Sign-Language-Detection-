import cv2
import pickle
import mediapipe as mp
import numpy as np

#load the model that we trained before
model_dict=pickle.load(open('./model.p','rb'))
model=model_dict['model']

cap=cv2.VideoCapture(0)
x1=0
y1=0
x2=0
y2=0
predicted_character=''

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#detect all the landmarks in the hand

labels_dict= {0: 'A', 1 : 'B', 2:'C', 3: 'D', 4: 'F',5: 'G',6:'H',7:'I',8:'L',9 : 'O',10:'P',11: 'Q',12:'U',13:'V',14: 'W'}

while True : 

    data_aux=[]
    x_=[]
    y_=[]
   
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    H, W, _ = frame.shape #the width and the height of the frame 

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks: 
       # for hand_landmarks in results.multi_hand_landmarks: #iterate in all the landmarks
           # mp_drawing.draw_landmarks(
            #    frame,  # image to draw
             #   hand_landmarks,  # model output
              #  mp_hands.HAND_CONNECTIONS,  # hand connections
               # mp_drawing_styles.get_default_hand_landmarks_style(),
                #mp_drawing_styles.get_default_hand_connections_style())
        
        for hand_landmarks in results.multi_hand_landmarks: #iterate in all the landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                    #normalization and build an array with the values of x and y of the landmarks position
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        #the corners of the rectangle containing the hand so we know where to display this rectangle and the text

        prediction = model.predict([np.asarray(data_aux)]) #this will predict from landmarks on the camera 
        #prediction is a list of only one element and thats why 0 
        predicted_character = labels_dict[int(prediction[0])] 
        #classify the class for the landmarks 
        print (predicted_character)

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0),4)

    cv2.putText(frame, predicted_character, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame',frame)
    cv2.waitKey(10) #wait 10ms between each frame

#detect all the landmarks in the hand
#use the classifier we trained to know which symbol i am displaying in my hand 


cap.release()
cv2.destroyAllWindows()

