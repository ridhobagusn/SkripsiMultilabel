import cv2
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
respon_warna = {
    'Tipe Kendaraan': 1,
    'Warna Kendaraan': 2,
    'Jenis Kendaraan': 3,
    'Tipe + Warna': 4,
    'tipe + Jenis': 5,
    'warna + jenis':6,
    'Semua':7
}

print(respon_warna)
# Meminta pengguna memasukkan pilihan1
pilihan = input("Masukkan pilihan Anda (pisahkan dengan koma): ")

# Membagi input menjadi list warna
pilihan_list = [warna.strip() for warna in pilihan.split(',')]
total_sum_strings = sum([int(num) for num in pilihan_list])
# Menampilkan pilihan yang dimasukkan pengguna

cap = cv2.VideoCapture(r'as.mp4')
net = cv2.dnn.readNetFromONNX(r"best (3).onnx")
classes = ["Kendaraan Umum","Kendaraan Pribadi","Motor","Kendaraan Berat"]
model = load_model(r"databaru.h5")

SIZE = 100
count = 0
umum = 0
pribadi = 0
berat = 0
frameSize = [1280, 720]
# cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_path = 'CCTV.mp4'
# out = cv2.VideoWriter(out_path,cv2_fourcc, cap.get(cv2.CAP_PROP_FPS), frameSize)


def draw_detection(img, r, color=(0, 255, 0), thickness=2):
    hor = r[2] // 7
    ver = r[3] // 7
    # Top left corner
    cv2.line(img, tuple(r[0:2]), (r[0], r[1] + ver), color, thickness)
    cv2.line(img, tuple(r[0:2]), (r[0] + hor, r[1]), color, thickness)
    # Top right corner
    cv2.line(img, (r[0] + r[2] - hor, r[1]), (r[0] + r[2], r[1]), color, thickness)
    cv2.line(img, (r[0] + r[2], r[1] + ver), (r[0] + r[2], r[1]), color, thickness)
    # Bottom right corner
    cv2.line(img, (r[0] + r[2], r[1] + r[3] - ver), (r[0] + r[2], r[1] + r[3]), color, thickness)
    cv2.line(img, (r[0] + r[2] - hor, r[1] + r[3]), (r[0] + r[2], r[1] + r[3]), color, thickness)
    # Bottom left corner
    cv2.line(img, (r[0], r[1] + r[3] - ver), (r[0], r[1] + r[3]), color, thickness)
    cv2.line(img, (r[0] + hor, r[1] + r[3]), (r[0], r[1] + r[3]), color, thickness)
def draw_text(image, teks, x1, y1, color, size=0.5):
    text_width, text_height = cv2.getTextSize(text=teks, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size,
                              thickness=2)[0]
    text_x = x1 + 5
    text_y = y1 - 5

    text_background_x1 = x1
    text_background_y1 = y1 - 2 * 5 - text_height

    text_background_x2 = x1 + 2 * 5 + text_width
    text_background_y2 = y1

    cv2.rectangle(image, pt1=(text_background_x1, text_background_y1), pt2=(text_background_x2, text_background_y2),
                  color=color, thickness=cv2.FILLED)
    contrast = generate_contrast_color(color)
    cv2.putText(image, text=teks, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=size,
                color=contrast, thickness=1, lineType=cv2.LINE_AA)
def generate_contrast_color(bgr_color):
    bgr_color = np.array(bgr_color)
    contrast_color = 255 - bgr_color
    return (int(contrast_color[0]), int(contrast_color[1]), int(contrast_color[2]))

while True:
    try:
        
        ret,img=cap.read()
        if not ret:
            break
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if pos_frame%19 == 0:
            img = cv2.resize(img, (1280, 720))
            blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
            net.setInput(blob)
            detections = net.forward()[0]
            # cx,cy , w,h, confidence, 80 class_scores
            # class_ids, confidences, boxes
            classes_ids = []
            confidences = []
            boxes = []
            rows = detections.shape[0]

            img_width, img_height = img.shape[1], img.shape[0]
            x_scale = img_width/640
            y_scale = img_height/640

            for i in range(rows):
                row = detections[i]
                confidence = row[4]
                if confidence > 0.2:
                    classes_score = row[5:]
                    ind = np.argmax(classes_score)
                    if classes_score[ind] > 0.2:
                        classes_ids.append(ind)
                        confidences.append(confidence)
                        cx, cy, w, h = row[:4]
                        x1 = int((cx- w/2)*x_scale)
                        y1 = int((cy-h/2)*y_scale)
                        width = int(w * x_scale)
                        height = int(h * y_scale)
                        box = np.array([x1,y1,width,height])
                        boxes.append(box)

            indices = cv2.dnn.NMSBoxes(boxes,confidences,0.2,0.2)
            # cv2.line(img, (0,400),(1280,400),(255,255,255),4)
            m="Jumlah Umum = " + str(umum)
            n="Jumlah Pribadi = " + str(pribadi)
            o="Jumlah Berat = " + str(berat)

            
            draw_text(img, m, 10, 80, (255,255,255), size=1)
            draw_text(img, n, 10, 130, (255,255,255), size=1)
            draw_text(img, o, 10, 180, (255,255,255), size=1)
            

            # cv2.putText(img, m, (10,80),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            # cv2.putText(img, n, (10,130),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            # cv2.putText(img, o, (10,180),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)

            
            
            # cv2.putText(img, str(umum), (350,80),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            # cv2.putText(img, str(pribadi), (350,130),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            # cv2.putText(img, str(berat), (350,180),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            for i in indices:
                x1,y1,w,h = boxes[i]
                imgg = img[y1:y1+h, x1:x1+w]
                label = classes[classes_ids[i]]
                TipeKendaraan = label
                imgg = cv2.resize(imgg, (SIZE, SIZE))
                imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV)
                imgg = (imgg/255)
                imgg = np.expand_dims(imgg, axis=0)
                classess = ['Biru','Hijau','Hitam','Merah','Putih','SUV','MPV','Sedan','Modern','Klasik','Toyota','Honda'] #Get array of all classes

                proba = model.predict(imgg) 
                print(proba)
                

                class_warna=['Biru','Hijau','Hitam','Merah','Putih']
                class_tipe=["SUV", "MPV",'Sedan']
                class_klasikmodern = ['Modern','Klasik']
                class_merk = ['Toyota','Honda']
                
                warna = proba[0][:5]
                tipe = proba[0][5:8]
                klasikmodernn = proba[0][8:10]
                merk = proba[0][10:]
             
                WarnaKendaraan = class_warna[np.argmax(warna)]
                JenisKendaraan = class_tipe[np.argmax(tipe)]
                klasikmodern = class_klasikmodern[np.argmax(klasikmodernn)]
                merkkendaraan = class_merk[np.argmax(merk)]
                
                offset = 5
                if (y1+h<(400+offset) and y1+h>(400-offset)):
                
                    if TipeKendaraan == classes[0]:
                        pribadi += 1
                    if TipeKendaraan == classes[1]:
                        umum += 1
                    if TipeKendaraan == classes[3]:
                        berat += 1


                



                # class_warna=["Putih","Hijau","Merah","Hitam"]
                # class_merk=["Toyota", "Honda"]
                # warna = proba[:len(classess)]
                # merk = proba[len(class_warna):len(class_warna)+len(class_merk)]
                # WarnaKendaraan = classess[warna.index(max(warna))]
                # merk_warna = class_merk[merk.index(max(warna))]
                if not label == "Motor":

                    cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),1)
                    draw_detection(img,(x1,y1,w,h))
                # cv2.putText(img, TipeKendaraan, (x1,y1+h+15),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,255,255),1)
                # cv2.putText(img, WarnaKendaraan, (x1,y1+h+33),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,255,255),1)
                # cv2.putText(img, JenisKendaraan, (x1,y1+h+50),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,255,255),1)
                    if total_sum_strings == 1:  
                        draw_text(img, TipeKendaraan, x1, y1-15, (0,255,0))
                        
                    if total_sum_strings == 2:
                        draw_text(img, WarnaKendaraan, x1, y1-15, (0,255,0))
                    if total_sum_strings == 3:
                        draw_text(img, JenisKendaraan, x1, y1-15, (0,255,0))
                    if total_sum_strings == 4:
                        draw_text(img, TipeKendaraan, x1, y1-33, (0,255,0))
                        draw_text(img, WarnaKendaraan, x1, y1-15, (0,255,0))
                    if total_sum_strings == 5:
                        if label == "Kendaraan Berat":
                            draw_text(img, TipeKendaraan, x1, y1-33, (0,255,0))
                        else:
                            draw_text(img, TipeKendaraan, x1, y1-33, (0,255,0))
                            draw_text(img, JenisKendaraan, x1, y1-15, (0,255,0))
                        
                    if total_sum_strings == 6:
                        if label == "Kendaraan Berat":
                            draw_text(img, WarnaKendaraan, x1, y1-33, (0,255,0))
                        else:
                            draw_text(img, WarnaKendaraan, x1, y1-33, (0,255,0))
                            draw_text(img, JenisKendaraan, x1, y1-15, (0,255,0))
                        
                    if total_sum_strings == 7:
                        if label == "Kendaraan Berat":
                            draw_text(img, TipeKendaraan, x1, y1-50, (0,255,0))
                        else:
                            draw_text(img, TipeKendaraan, x1, y1-50, (0,255,0))
                            draw_text(img, WarnaKendaraan, x1, y1-15, (0,255,0))
                            draw_text(img, JenisKendaraan, x1, y1-33, (0,255,0))
                            draw_text(img, klasikmodern, x1, y1-67, (0,255,0))
                            draw_text(img, merkkendaraan, x1, y1-94, (0,255,0))

            cv2.imshow("CCTV",img)
            # out.write(img)
            if cv2.waitKey(1)&0xFF==27:
                break        


    except Exception as e:
        print(e)

# out.release()
