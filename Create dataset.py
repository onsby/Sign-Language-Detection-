import os
import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt
import pickle

mp_hands= mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# to draw the landmarks on the image 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# define the object (the model )
DATA_DIR='./data'

data=[] #in order to make the classification
labels=[] # the category of each one image(each data point)

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux=[]  #for each image we create an array
        x_ = []
        y_ = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb= cv2.cvtColor(img,cv2.COLOR_BGR2RGB )

        results = hands.process(img_rgb)
        #detect the landmarks in the hand
        if results.multi_hand_landmarks : #becauce it can detect one hand more or no one so we make sure that if it detect one hand so .. 
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)) : #the values of the landmarks on x y and z 
                    x= hand_landmarks.landmark[i].x
                    y= hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                    #normalization and build an array with the values of x and y of the landmarks position
            data.append(data_aux) #the positions values
            labels.append(dir_) #the category name(number)
            #creating our dataset in order to train our classifier 
        #iterate in the landmarks that we detected in the image (one by one )
f=open('data.pickle','wb') #open a file to save the data
pickle.dump({'data' : data , 'labels' : labels},f ) #create a dictionnary with the dataset in the file 

f.close()