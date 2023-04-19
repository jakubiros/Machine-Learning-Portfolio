'''

Program do automatycznej i szybkiej obróbki zdjęć z OneDrive wykorzystujący kilka wytrenowanych sieci konwolucyjnych do poprawy jakości obrazu. Program głównie działa na raspberrypi podłączonym do sieci i chmury OneDrive. OneDrive jest zmontowany na raspberrypi przez biblioteke rclone. Na wejściu program szuka w folderze zdjęcia do obórbki. Następnie zdjęcie jest przetwarzane przez sieć do klasyfikacji która rozróżnia 7 różnych scen(Krajobraz,górski,miejski,leśny,wodny,wschód/zachód,nocny). Po określeniu klasy, zdjęcie jest poddane podstawowej obróbce(jasność,kontrast,nasycenie,wyostrzanie,odszumianie)unikalnej dla każdej klasy. Następnie zdjęcie jest odszumiane wytrenowaną do tego siecią U-Net. Potem w zależności od wymiarów zdjęcie jest powiększane siecią SRGAN 2-krotnie lub 4-krotnie. Na końcu zdjęcie jest zapisywane i usuwane z folderu wejściowego, aby zwalniać miejsce na kolejne.

'''
#Importowanie potrzebnych bibliotek
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import gc
import random
import time
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from wand.image import Image
from datetime import date
from PIL import Image as pil
from PIL import ImageEnhance as ie
from tensorflow import keras
from tensorflow.keras import backend as K
from skimage.io import imsave
from time import sleep
from skimage import exposure

#Ustalenie ilości używanych wątków procesora podczas wykonywania obliczeń dla sieci, aby zredukować ilość alokowanej pamięci RAM
K.clear_session()
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(4)

input_path='/home/pi/OneDrive/Auto_Process/Input/'
output_path='/home/pi/OneDrive/Auto_Process/Output/'

#Wczytuje zdjęcie różnych typów(jpg,png,nef,dng)
def img_loader(img_path,filename):
    if '.jpg' in filename or '.png' in filename:
        img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        return img,'noraw'
    elif '.NEF' in filename or '.dng' in filename or '.DNG' in filename or '.nef' in filename:
        img=raw_preprocess(img_path)
        return img,'raw'

#Funkcja wczytująca i wyostrzająca wstępnie zdjęcia w formacie RAW, 
def raw_preprocess(img_array):
    img=rawpy.imread(img_array)
    img=img.postprocess(output_bps=8,output_color=rawpy.ColorSpace.sRGB,fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(1),dcb_enhance=True,chromatic_aberration=(1,1))
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_s=sharpen_conv(img)
    img_t=layer_thresh(img,img_s,0.65)
    return img_t

#Wyostrzanie poprzez filtrowanie zdjęcia 
def sharpen_conv(img):
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpen=cv2.filter2D(img,-1,kernel)
    return sharpen

#W zależności od typu zdjęcia czy jest to raw lub nie, wstępnie wyostrza oraz zwiększa nasycenie kolorów.
def entry_image_postprocess(img_array,img_type):
    if img_type=='noraw':
        img=pil.fromarray(img_array)
        enh=ie.Sharpness(img)
        img=enh.enhance(1.4)
        enh2=ie.Color(img)
        img=enh2.enhance(1.3)
        return np.array(img)
    elif img_type=='raw':
        img=pil.fromarray(img_array)
        enh=ie.Sharpness(img)
        img=enh.enhance(1.5)
        enh2=ie.Color(img)
        img=enh2.enhance(1.6)
        return np.array(img)
    
#Funkcja która pozwala na dostosowanie przeźroczystości dwóch zdjęć.
def layer_thresh(img1,img2,alpha):
    beta=(1.0-alpha)
    result=cv2.addWeighted(img1,alpha,img2,beta,0.0)
    return result

#Funkcja wykorzystująca wcześniej wytrenowaną sieć do klasyfikacji obrazów
def scene_predict(img_array):
    model=load_model('/home/pi/FTPR4.h5',custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})
    img_to_pred=cv2.resize(img_array,(480,416))
    img_to_pred=img_to_pred.astype('float32')/255
    img_to_pred=np.expand_dims(img_to_pred,axis=0)
    prediction=model.predict(img_to_pred)
    pred=np.argmax(prediction)
    #Zwalnianie niepotrzebnych zasobów i zmiennych
    K.clear_session()
    del model
    gc.collect()
    return pred
#korzystanie wyłącznie z procesora urządzenia
with tf.device('/CPU:0'):

    #Funkjce służące do dzielenia zdjęcia na cztery równe części oraz ponownemu łączeniu ich w celu uniknięcia błędów z alokacją pamięci podczas korzystania z sieci do odszumiania i interpolacji.
    def img_patcher(img_array,num_patches='4'):
        img=cv2.cvtColor(img_array,cv2.COLOR_BGR2YUV)
        x=div_check(img.shape[1])
        y=div_check(img.shape[0])
        img=resizer(img,x,y)
        h,w,c=img.shape
        if num_patches=='4':
            patch1=img[0:int(h/2),0:int(w/2),0]
            patch2=img[0:int(h/2),int(w/2):w,0]
            patch3=img[int(h/2):h,0:int(w/2),0]
            patch4=img[int(h/2):h,int(w/2):w,0]
            return (patch1,patch2,patch3,patch4),img
        else:
            return img

        def img_concat(img_patches):
        l=[]
        temp=[]
        amount=len(img_patches)
        for i in range(amount):
            horizontal_patches=img_patches[i]
            temp.append(horizontal_patches)
            if i%2==1:
                horizontal=np.concatenate((temp),axis=1)
                l.append(horizontal)
                temp=[] 
        vertical=np.concatenate((l),axis=0)
        return vertical
    
    #Funkcja odpowiadająca za odszumianie oraz ewentualne usunięcie blokowści jpeg zdjęcia wejściowego w celu poprawienia jego jakości. Odbywa się to za pomocą wytrenowanej małej sieci U-net techniką GAN
    def denoise2(img_array):
        denoiser=load_model('/home/pi/DNGEN90pg.h5',custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU},compile=False)
        img_patches,full_img=img_patcher(img_array,num_patches='4')
        full_img=full_img.astype('float32')/255
        pred_patches=[]
        for i in img_patches:
            img=np.expand_dims(i/255,axis=0)
            img_pred=denoiser.predict(img)
            pred_patches.append(img_pred)
        score=np.array(pred_patches)
        score=score[:,0,:,:,:]
        score=img_concat(score)
        full_img[:,:,0]=score[:,:,0]
        final=(full_img*255).astype('uint8')
        final=cv2.cvtColor(final,cv2.COLOR_YUV2BGR)
        del denoiser
        K.clear_session()
        gc.collect()
        return final
    #Podobnie jak funkcja wyżej odszumia zdjęcie wejściowe za pomocą sieci U-net wytrenowanej na zdjęciach ze słabymi warunkami świetlnymi i jest używana tylko kiedy sieć klasyfikiacyjna przewidzi scenerię nocną.
    def night_denoise(img_array):
        K.clear_session()
        denoiser_night=load_model('/home/pi/models/NVHRrasp8U.h5')
        n=np.array(n)/255.
        n=n.astype('float32')
        img=denoiser_night.predict(n)
        img=(img*255).astype(np.uint8)
        return img
    #Funkcja, której zadaniem jest wczytanie odpowiedniego modelu sieci i zwiększenie natywnej rozdzielczości zdjęcia(2x,4x).
    #Więcej szczegółów w pliku SRGAN.ipynb    
    def SRGAN(img_array,mode):
        if mode=='4x':
            gan=load_model('/home/pi/SRmodels/GEN200v3-4x.h5',compile=False,custom_objects={'LeakyReLU':tf.keras.layers.LeakyReLU})
        else:
            gan=load_model('/home/pi/SRmodels/GEN190v3-2x.h5',compile=False,custom_objects={'LeakyReLU':tf.keras.layers.LeakyReLU})
        img=cv2.cvtColor(img_array,cv2.COLOR_RGB2BGR)
        img=img.astype('float32')/127.5-1
        img=np.expand_dims(img,axis=0)
        score=gan.predict(img)
        score=score[0,:,:,:]
        score=cv2.cvtColor(score,cv2.COLOR_BGR2RGB)
        score=((score+1)*127.5).astype(np.uint8)
        final=histogram_match(img_array,score)
        return final
        del gan
        K.clear_session()
        gc.collect()
#Funkcja pozwalająca na dopasowanie histogramów dwóch tych samych zdjęć, ale z różnym zakresem tonalnym.   
    def histogram_match(img1,img2):
        y,x,c=img2.shape
        img1=cv2.resize(img1,(x,y),interpolation=cv2.INTER_CUBIC)
        matched=exposure.match_histograms(img2,img1,channel_axis=-1)
        return matched
        
#Zbiór funkcji retuszu zdjęć dla siedmiu scenerii. Zawierają one kilka parametrów do poprawy obrazu głownie takie jak gamma, nasycenie,  czy wyostrzanie. Są to drobne poprawki, aby nie pogorszyć jakości zdjęcia wejściowego.
def gory(img_array):
    img=Image.from_array(img_array)
    img.gamma(1.07)
    img.colorize(color='green',alpha='rgb(3%,4%,5%)')
    img.adaptive_sharpen(radius=5, sigma=3)
    img.enhance()
    img=np.array(img)
    return img   

def miejski(img_array):
    img=Image.from_array(img_array)
    img.level(0.1,0.95,gamma=1.05)
    img.adaptive_sharpen(radius=5,sigma=4)
    img.enhance()
    img=np.array(img)
    return img

def lesny(img_array):
    img=Image.from_array(img_array)
    img.gamma(1.08)
    img.adaptive_blur(radius=3, sigma=2)
    img.adaptive_sharpen(radius=4, sigma=3)
    img.colorize(color='green',alpha='rgb(3%,5%,10%)')
    img.enhance()
    img=np.array(img)
    return img

def krajobraz(img_array):
    img=Image.from_array(img_array)
    img.gamma(1.06)
    img.adaptive_blur(radius=3, sigma=1)
    img.adaptive_sharpen(radius=2, sigma=1)
    img.adaptive_sharpen(radius=4, sigma=2)
    img.enhance()
    img=np.array(img)
    return img

def wodny(img_array):
    img=Image.from_array(img_array)
    img.gamma(1.1)
    img.sharpen(radius=4,sigma=2)
    img.colorize(color='green',alpha='rgb(3%,7%,16%)')
    img.adaptive_blur(radius=3, sigma=2)
    img.colorize(color='green',alpha='rgb(3%,7%,16%)')
    img.enhance()
    img=np.array(img)
    return img

def wsch(img_array):
    img=Image.from_array(img_array)
    img.gamma(1.15)
    img.sharpen(radius=4,sigma=2)
    img.adaptive_blur(radius=3, sigma=2)
    img.colorize(color='red',alpha='rgb(5%,7%,3%)')
    img.enhance()
    img=np.array(img)
    return img

def nocny(img_array):
    img=Image.from_array(img_array)
    img.gamma(1.05)
    img.adaptive_blur(radius=3, sigma=2)
    img.gaussian_blur(sigma=1)
    img.adaptive_sharpen(radius=5, sigma=2)
    img.enhance()
    img=np.array(img)
    return img

#Decudyje, która funkcja z klas ma się wykonać w zależności od wyników sieci klasyfikacyjnej.
def scene_postprocess(nr,img_array):
    func_list=[gory,wodny,miejski,wsch,lesny,krajobraz,nocny]
    img=func_list[nr](img_array)
    img=denoise2(img)
    return img

#Implementacja CLAHE (Contrast Limited Adaptive Histogram Equalization), czyli globalna normalizacja histogramu.
def histogram_norm(img_array,amount):
    img=cv2.cvtColor(img_array,cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img[:,:,0] = clahe.apply(img[:,:,0])
    img=cv2.cvtColor(img,cv2.COLOR_LAB2RGB)
    thresh_img=layer_thresh(img_array,img,amount)
    return thresh_img

#Sprawdzanie czy jest konieczność użycia sieci do interpolacji
def shape_check_srgan(img_array):
    h,w,c=img_array.shape
    if w<1120 and h<960:
        image=SRGAN(img_array,'4x')
        return image
    elif w>1120 and w<2100 and h>960 and h<1700:
        image=SRGAN(img_array,'2x')
        return image
    else:
        return img_array
#Funkcja sprawdzająca wymiary zdjęcia wejściowego, czy jest podzielne przez 5 razy.    
def div_check(x):
    while True:
        if int(x)%32==0:
            return x
            break
        else:
            x=x-1
#Wyodrębnianie nazwy zdjęcia    
def img_name(name):
    if 'NEF' in name:
        out=name.split('.')[0]
        return out
    else:
        todaydate=date.today()
        d1=todaydate.strftime('%d%m%Y')
        t=time.localtime()
        ct=time.strftime('%H%M%S',t)
        out='IMG'+str(d1)+'_'+str(ct)
        return out
#Funkcja zmieniająca wymiary zdjęcia
def resizer(img_array,x,y):
        img_array=cv2.resize(img_array,(x,y),interpolation=cv2.INTER_AREA)
        return img_array

#Funkcja zapisująca zdjęcie w różnych foramtach.
def img_saver(img_array,name,input_img_name):
    img_formats=['.png','.jpg']
    if '.NEF' in input_img_name or '.dng' in input_img_name or '.png' in input_img_name:
        img_type=img_formats[0]
    elif '.jpg' in input_img_name or '.jpeg' in input_img_name:
        img_type=img_formats[1]
    else:
        img_type=img_formats[1]
    cv2.imwrite(output_path+name+img_type,img_array)
    

input_img='calibration_field'

#Pętla przeszukująca folder w poszukiwaniu zdjęć
while True:
    try:
        input_img=random.choice(os.listdir(input_path))
        img_path=input_path+input_img
    except IndexError:
    if os.path.exists(input_path+input_img) is True:
        img_path=input_path+input_img
        break
    else:
        sleep(10)
        continue

img,img_format=img_loader(img_path,input_img)
img=entry_image_postprocess(img,img_format)
scene_nr=scene_predict(img)
img=scene_postprocess(scene_nr,img)
img=histogram_norm(img,0.5)
img=shape_check_srgan(img)
name=img_name(input_img)
img_saver(img,name,input_img)
#Usunięcie zdjęcia z folderu  
os.remove(img_path)
#Ponowne uruchomienie programu
os.system('python3 /home/pi/PRauto2.py')



