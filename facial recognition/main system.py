import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

'''in this we are loading images from the folder we are not writing the code for each image to be loaded'''
folder_path = 'images'
images = []
names = []
contents = os.listdir(folder_path)  # listdir returns the name of the file
# print(contents)
for cl in contents:
    # for loop is use to read  each image in the folder
    current_image = cv2.imread(f'{folder_path}/{cl}')
    '''{folder_path}/{cl} means folder_path is image and cl is contents 
    i.e first iterartion will be images/arsh.jpg and second iterarion will b
    e images/elon musk and so on... 
    till all the images are readed'''
    images.append(current_image)
    # .appened is used to add something in the images variable which is empty
    names.append(os.path.splitext(cl)[0])
    '''removing the image extention i.e(.jpg) by using (os.path.splitext) 
    and [0] is used to take 1st word of splitted text'''


# function for encoding images
def encodings(images):
    print("encoding starts....")
    encode_lists = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_lists.append(encode)
    return encode_lists


# get date time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H:%M")
first_line = "NAME,TIME,DATE"
with open(f'attendance records/{current_date}.csv', 'w') as f:
    f.write(first_line)


def mark_attendence(name):
    date = now.strftime("%m-%d")
    with open(f'attendance records/{current_date}.csv', 'r+') as a:
        data_list = a.readlines()
        # print(data_list)
        name_list = []
        for lines in data_list:
            entry = lines.split(',')  # splits the  line into list of elements using the comma as a separator
            name_list.append(entry[0])  # entry [0] will be name it is first data element written in  file
        if name not in name_list:
            a.writelines(f'\n{name},{current_time},{date}')


known_encoding = encodings(images)
print("encoding completed")
#
# video capturing
print("starting camera...\n This will take few seconds")
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,900)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,1200)

while True:
    success, frame = cap.read()
    # it will reduce the size of img by (1/4th=0.25) to speed up the process
    imge = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)

    # detect face and encodings
    current_face_locatn = face_recognition.face_locations(imge)
    current_face_encoding = face_recognition.face_encodings(imge, current_face_locatn)

    for enface, faceloc in zip(current_face_encoding, current_face_locatn):
        match = face_recognition.compare_faces(known_encoding, enface)
        face_distance = face_recognition.face_distance(known_encoding, enface)
        matchindex = np.argmin(face_distance)  # it returns the minimum value
        if match[matchindex]:
            name = names[matchindex]
            print(f"Attendance mark for {name} at {current_time}")   #this will show the name
            top, right, bottom, left = faceloc
            top, right, bottom, left = top*2 , right*2 , bottom*2 , left*2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            mark_attendence(name)

    cv2.imshow("face recognition", frame)
    # break the loop if pressed key is esc
    t = cv2.waitKey(1)
    if t == 27:
        break
cap.release()
cv2.destroyAllWindows()
