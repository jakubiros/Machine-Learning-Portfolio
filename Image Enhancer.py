'''

Skrypt do automatycznej i szybkiej obróbki zdjęć z OneDrive wykorzystujący kilka wytrenowanych sieci konwolucyjnych do poprawy jakości obrazu.
Skrypt głównie działa na raspberrypi podłączonym do sieci i chmury OneDrive. OneDrive jest zmontowany na raspberrypi przez biblioteke rclone.
Na wejściu program szuka w folderze zdjęcia do obórbki. Następnie zdjęcie jest przetwarzane przez sieć do klasyfikacji która rozróżnia 7 różnych scen(Krajobraz,górski,miejski,leśny,wodny,wschód/zachód,nocny).
Po określeniu klasy, zdjęcie jest poddane podstawowej obróbce(jasność,kontrast,nasycenie,wyostrzanie,odszumianie)unikalnej dla każdej kategorii. Następnie zdjęcie jest odszumiane wytrenowaną do tego siecią.
Potem w zależności od wymiarów zdjęcie jest powiększane siecią SRGAN 2-krotnie lub 4-krotnie. Na końcu zdjęcie jest zapisywane i usuwane z folderu wejściowego, aby zwalniać miejsce na kolejne.

'''

import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import gc
import random
import time
import skimage
import matplotlib.image as mpimg
import tensorflow as tf
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
from tensorflow.python.eager import context

os.environ["TF_OVERRIDE_GLOBAL_THREADPOOL"] = "1"

#Ustalenie ilości używanych rdzeni i wątków procesora podczas wykonywania obliczeń dla sieci, aby zredukować ilość alokowanej pamięci RAM
class ConfigThreading:
    __instance=None
    def __new__(cls,*args,**kwargs):
        if not cls.__instance:
            cls.__instance=super().__new__(cls)
        return cls.__instance
    def __init__(self):
        self.cores=4
        self.threads=4
    
    def set_config(self,cores=4,threads=4):
        K.clear_session()
        self.cores=cores
        self.threads=threads
        context._reset_context()
        tf.config.threading.set_intra_op_parallelism_threads(cores)
        tf.config.threading.set_inter_op_parallelism_threads(threads)

#Klasa odpowiadająca za wczytywanie zdjęcia i jego wstępny retusz
class Load_img:
    def __init__(self, img_path, filename):
        self.img_path = img_path
        self.filename = filename
        self.img = None
        self.img_type = None
        
    #Funkcja wczytująca i wyostrzająca wstępnie zdjęcia w formacie RAW,
    def raw_preprocess(self):
        self.img = rawpy.imread(self.img_path)
        self.img = self.img.postprocess(output_bps=8, output_color=rawpy.ColorSpace.sRGB, fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(
            2), dcb_enhance=True, chromatic_aberration=(1, 1),use_auto_wb=True,median_filter_passes=2)
        self.img = np.array(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_s = self.sharpen_conv(self.img)
        self.img = self.layer_thresh(img_s,self.img, 0.15)
        return self.img

    @staticmethod
    def sharpen_conv(img):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(img, -1, kernel)
        return sharpen
        
    #Funkcja która pozwala na dostosowanie przeźroczystości dwóch zdjęć.
    @staticmethod
    def layer_thresh(img1, img2, alpha):
        beta = (1.0-alpha)
        result = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
        return result
    
    #Szacuje ostrość zdjęcia
    @staticmethod
    def sharpness_metric(img):
        g=np.gradient(img)
        gnorm=np.sqrt(g[1]**2+g[0]**2)
        sharpness=np.average(gnorm)
        return sharpness
    
    #Szacuje odchylenie standardowe szumu 
    @staticmethod
    def estimate_noise(img):
        img=skimage.restoration.estimate_sigma(img, average_sigmas=True, channel_axis=-1)
        return img
    
    #Wczytuje zdjęcie różnych typów(jpg,png,nef,dng)
    def img_loader(self):
        if '.jpg' in self.filename or '.png' in self.filename:
            self.img_type = 'noraw'
            self.img=cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
            if self.img.dtype=='uint16':
                self.img=np.float32(self.img)
                self.img /=65535
                self.img *=255
                self.img=np.clip(self.img,0,255)
                self.img=self.img.astype('uint8')
            return self.img
        elif '.NEF' in self.filename or '.dng' in self.filename or '.DNG' in self.filename or '.nef' in self.filename:
            self.img_type = 'raw'
            self.img=self.raw_preprocess()
            return self.img
            
    #W zależności od typu zdjęcia czy jest to raw lub nie, wstępnie wyostrza oraz zwiększa nasycenie kolorów.
    def entry_image_postprocess(self):
        if self.img_type == 'noraw':
            self.img = pil.fromarray(self.img)
            enh = ie.Sharpness(self.img)
            self.img = enh.enhance(1.05)
            enh2 = ie.Color(self.img)
            self.img = enh2.enhance(1.40)
            return np.array(self.img)
        elif self.img_type == 'raw':
            self.img = pil.fromarray(self.img)
            enh = ie.Sharpness(self.img)
            self.img = enh.enhance(1.1)
            enh2 = ie.Color(self.img)
            self.img = enh2.enhance(1.6)
            return np.array(self.img)

    def entry_pipeline(self):
        self.img=self.img_loader()
        self.img=self.entry_image_postprocess()
        return self.img

#Klasa wykorzystująca wcześniej wytrenowaną sieć do klasyfikacji obrazów
class Predict_scene:
    def __init__(self, model_path,img):
        self.model_path = model_path
        self.img=img

    def scene_predict(self):
        model = load_model(self.model_path)
        img_to_pred = cv2.resize(self.img, (384,256))
        img_to_pred=tf.keras.applications.resnet_v2.preprocess_input(img_to_pred)
        img_to_pred = np.expand_dims(img_to_pred, axis=0)
        prediction = model.predict(img_to_pred)
        pred = np.argmax(prediction)
        K.clear_session()
        del model
        gc.collect()
        return pred

#Zbiór funkcji retuszu zdjęć dla siedmiu scenerii. Zawierają one kilka parametrów do poprawy obrazu głownie takie jak gamma, nasycenie,
#czy wyostrzanie. Są to drobne poprawki, aby nie pogorszyć jakości zdjęcia wejściowego.
class Scene_postprocess:
    def __init__(self,img,scene_nr):
        self.img=img
        self.scene_nr=scene_nr
        self.gradient=Load_img.sharpness_metric(self.img)
        self.amount=self.scene_sharpen(self.scene_nr)

    def postprocess(self):
        func_list = [self.gory, self.wodny, self.miejski,
                     self.wsch, self.lesny, self.krajobraz, self.nocny]
        self.img = func_list[self.scene_nr]()
        self.img=self.white_balance_correction()
        self.img=self.sharpen_image(self.img,self.amount)
        return self.img

    def gory(self):
        self.img = Image.from_array(self.img)
        self.img.gamma(1.1)
        self.img.colorize(color='green', alpha='rgb(3%,4%,5%)')
        self.img.enhance()
        self.img = np.array(self.img)
        return self.img

    def miejski(self):
        self.img = Image.from_array(self.img)
        self.img.level(0.1, 0.95, gamma=1.05)
        self.img.enhance()
        self.img = np.array(self.img)
        return self.img

    def lesny(self):
        self.img = Image.from_array(self.img)
        self.img.gamma(1.08)
        self.img.adaptive_blur(radius=2, sigma=1)
        self.img.colorize(color='green', alpha='rgb(3%,5%,10%)')
        self.img.enhance()
        self.img = np.array(self.img)
        return self.img

    def krajobraz(self):
        self.img = Image.from_array(self.img)
        self.img.gamma(1.08)
        self.img.enhance()
        self.img = np.array(self.img)
        return self.img

    def wodny(self):
        self.img = Image.from_array(self.img)
        self.img.gamma(1.1)
        self.img.colorize(color='green', alpha='rgb(3%,7%,16%)')
        self.img.adaptive_blur(radius=2, sigma=1)
        self.img.colorize(color='green', alpha='rgb(3%,7%,16%)')
        self.img.enhance()
        self.img = np.array(self.img)
        return self.img

    def wsch(self):
        self.img = Image.from_array(self.img)
        self.img.gamma(1.17)
        self.img.adaptive_blur(radius=2, sigma=1)
        self.img.colorize(color='red', alpha='rgb(5%,7%,3%)')
        self.img.enhance()
        self.img = np.array(self.img)
        return self.img

    def nocny(self):
        self.img = Image.from_array(self.img)
        self.img.gamma(1.05)
        self.img.adaptive_blur(radius=3, sigma=2)
        self.img.enhance()
        self.img = np.array(self.img)
        return self.img
    
    #Korekcja balansu bieli
    def white_balance_correction(self):
        self.img=cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        avg_a = np.average(self.img[:, :, 1])
        avg_b = np.average(self.img[:, :, 2])
        self.img[:, :, 1] = self.img[:, :, 1] - ((avg_a - 128) * (self.img[:, :, 0] / 255.0) * 1.2)
        self.img[:, :, 2] = self.img[:, :, 2] - ((avg_b - 128) * (self.img[:, :, 0] / 255.0) * 1.2)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2RGB)
        return self.img
        
    #Funkcja liniowa obliczająca ilość wyostrzania
    def sharpen_value_linear(self,a,x,b):
        return round((a*x+b)/100,4)
    
    #Metody wysotrzające dwoma różnymi kernelami, które wyostrzają zdjęcie
    @staticmethod
    def sharpen_image(img_arr,amount):
        if amount<=0:
            return img_arr
        else:
            img=np.float64(img_arr)/255
            kernel1=np.array([[ 0.,  1.,  0.],[ 1., -4.,  1.],[ 0.,  1.,  0.]],np.float32)
            sharpen1=cv2.filter2D(img,cv2.CV_64F,kernel1)
            sharpen1=np.abs(img-sharpen1)
            sharpen1[sharpen1>1.0]=1.0
            sharpen1[sharpen1<0.0]=0.0
            sharpen1=(sharpen1*255).astype(np.uint8)
            kernel2=np.array([[-1., -1., -1.],[-1.,  8., -1.],[-1., -1., -1.]],np.float32)
            sharpen2=cv2.filter2D(img,cv2.CV_64F,kernel2)
            sharpen2=np.abs(img+sharpen2)
            sharpen2[sharpen2>1.0]=1.0
            sharpen2[sharpen2<0.0]=0.0
            sharpen2=(sharpen2*255).astype(np.uint8)
            add_sharpened=Load_img.layer_thresh(sharpen1,sharpen2,0.5)
            final_sharpen=Load_img.layer_thresh(add_sharpened,img_arr,amount)
            return final_sharpen
        
    def scene_sharpen(self,nr):
        if nr==1 or nr==3:
            self.amount=self.sharpen_value_linear(-14,self.gradient,100)
            return self.amount
        elif nr==6:
            self.amount=self.sharpen_value_linear(-15,self.gradient,100)
            return self.amount
        else:
            self.amount=self.sharpen_value_linear(-13,self.gradient,100)
            return self.amount
            
#Klasa odpowiadająca za odszumianie, usunięcie blokowści jpeg oraz artefaktów zdjęcia wejściowego w celu poprawienia jego jakości. Odbywa się to za pomocą wytrenowanej małej, szybkiej sieci MFDNet. #https://arxiv.org/abs/2211.04687
class Denoise_img:
    def __init__(self,img ,model_path, norm_amount=0.5):
        self.img=img
        self.model_path = model_path
        self.norm_amount = norm_amount
        self.noise_amount=Load_img.estimate_noise(self.img)
        self.denoise_amount=self.denoise_linear_value(25,self.noise_amount,0)
        
    #Funkcja sprawdzająca wymiary zdjęcia wejściowego, czy jest podzielne przez 5 razy. 
    def div_check(self,x):
        while True:
            if int(x) % 32 == 0:
                return x
                break
            else:
                x = x-1
                
    #Funkcja liniowa obliczająca ile zastosować odszumiania
    def denoise_linear_value(self,a,x,b):
        if x>4:
            x=4
        return round((a*x+b)/100,4)
        
    def resizer(self, x, y):
        self.img = cv2.resize(
        self.img, (x, y), interpolation=cv2.INTER_AREA)
        return self.img
        
    #Funkjce służące do dzielenia zdjęcia na cztery równe części oraz ponownemu łączeniu ich w celu uniknięcia błędów z alokacją pamięci podczas korzystania z sieci do odszumiania i interpolacji.
    def img_patcher(self, num_patches='4'):
        x = self.div_check(self.img.shape[1])
        y = self.div_check(self.img.shape[0])
        self.img = self.resizer(x, y)
        h, w, c = self.img.shape
        if num_patches == '4':
            patch1 = self.img[0:int(h/2), 0:int(w/2)]
            patch2 = self.img[0:int(h/2), int(w/2):w]
            patch3 = self.img[int(h/2):h, 0:int(w/2)]
            patch4 = self.img[int(h/2):h, int(w/2):w]
            return (patch1, patch2, patch3, patch4)
        else:
            return self.img

    def img_concat(self,img_patches):
        l = []
        temp = []
        amount = len(img_patches)
        for i in range(amount):
            horizontal_patches = img_patches[i]
            temp.append(horizontal_patches)
            if i % 2 == 1:
                horizontal = np.concatenate((temp), axis=1)
                l.append(horizontal)
                temp = []
        vertical = np.concatenate((l), axis=0)
        return vertical

    def denoise(self):
        denoiser = load_model(self.model_path)
        img_patches = self.img_patcher(num_patches='4')
        img_ref=self.img.copy()
        pred_patches = []
        for i in img_patches:
            img = np.expand_dims(i/255, axis=0)
            img_pred = denoiser.predict(img)
            pred_patches.append(img_pred)
        score = np.array(pred_patches)
        score = score[:, 0, :, :, :]
        score = self.img_concat(score)
        score = (score*255)
        score = np.clip(score,0,255).astype('uint8')
        self.img=Load_img.layer_thresh(score,img_ref,self.denoise_amount)
        del denoiser
        K.clear_session()
        gc.collect()
        return self.img
        
    #Implementacja CLAHE (Contrast Limited Adaptive Histogram Equalization), czyli globalna normalizacja histogramu.
    def histogram_norm(self):
        img_ref=self.img.copy()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.img[:, :, 0] = clahe.apply(self.img[:, :, 0])
        self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2RGB)
        self.img = Load_img.layer_thresh(img_ref, self.img, self.norm_amount)
        return self.img
        
#Klasa, której zadaniem jest wczytanie odpowiedniego modelu sieci i zwiększenie natywnej rozdzielczości zdjęcia(2x,4x).
#Więcej szczegółów w pliku SRGAN.ipynb        
class Upscale_img:
    def __init__(self,img ,x2_model_path, x4_model_path):
        self.img=img
        self.x2_model_path = x2_model_path
        self.x4_model_path = x4_model_path
        self.upscaling_threading=ConfigThreading()
    
    #Funkcja pozwalająca na dopasowanie histogramów dwóch tych samych zdjęć, ale z różnym zakresem tonalnym. 
    def histogram_match(self,img1):
        y, x, c = self.img.shape
        img1 = cv2.resize(img1, (x, y), interpolation=cv2.INTER_CUBIC)
        matched = exposure.match_histograms(self.img, img1, channel_axis=-1)
        return matched

    def SRGAN(self, mode):
        self.upscaling_threading.set_config(1,1)
        if mode == '4x':
            gan = load_model(self.x4_model_path,custom_objects={"Addons>SpectralNormalization": tf.keras.layers.SpectralNormalization} ,compile=False)
        else:
            gan = load_model(self.x2_model_path, custom_objects={"Addons>SpectralNormalization": tf.keras.layers.SpectralNormalization}, compile=False)
        img_ref=self.img.copy()
        self.img = self.img.astype('float32')/255
        self.img = np.expand_dims(self.img, axis=0)
        self.img = gan.predict(self.img)
        self.img = self.img[0, :, :, :]
        self.img= np.clip(self.img,0,1)
        self.img = (self.img*255).astype(np.uint8)
        self.img = self.histogram_match(img_ref)
        del gan
        K.clear_session()
        gc.collect()
        self.upscaling_threading.set_config(4,4)
        return self.img

    #Sprawdzanie czy jest konieczność użycia sieci do interpolacji
    def srgan_pipeline(self):
        h, w, c = self.img.shape
        if w < 1120 and h < 960:
            self.img = self.SRGAN('4x')
            return self.img
        elif w > 1120 and w < 2100 and h > 960 and h < 1700:
            self.img = self.SRGAN('2x')
            return self.img
        else:
            return self.img

#Klasa zajmująca się dopasowaniem finalnej nnazwy zdjęcia
class Save_file:
    def __init__(self,img,filename,output_path):
        self.img=img
        self.filename=filename
        self.output_path=output_path
        self.name=None

    def img_name(self):
        if 'NEF' in self.filename:
            out = self.filename.split('.')[0]
            return out
        else:
            todaydate = date.today()
            d1 = todaydate.strftime('%d%m%Y')
            t = time.localtime()
            ct = time.strftime('%H%M%S', t)
            out = 'IMG'+str(d1)+'_'+str(ct)
            return out
            
    #Funkcja zapisująca zdjęcie w różnych foramtach.
    def img_saver(self):
        img_formats = ['.png', '.jpg']
        if '.NEF' in self.filename or '.dng' in self.filename or '.png' in self.filename:
            img_type = img_formats[0]
        elif '.jpg' in self.filename or '.jpeg' in self.filename:
            img_type = img_formats[1]
        else:
            img_type = img_formats[1]
        cv2.imwrite(self.output_path+self.name+img_type, self.img)
    
    def save_pipeline(self):
        self.name=self.img_name()
        self.img_saver()


def main():
    threading=ConfigThreading()
    threading.set_config(4,4)
    input_path = '/home/pi/OneDrive/Auto_Process/Input/'
    output_path = '/home/pi/OneDrive/Auto_Process/Output/'

    input_img='a'

    #Pętla przeszukująca folder w poszukiwaniu zdjęć
    while True:
        try:
            input_img=random.choice(os.listdir(input_path))
            img_path=input_path+input_img
            break
        except IndexError:
            if os.path.exists(input_path+input_img) is True:
                img_path=input_path+input_img
                break
            else:
                sleep(10)
                continue

    s=time.perf_counter()
    img_path = input_path+input_img

    loader=Load_img(img_path,input_img)
    img=loader.entry_pipeline()
    sc=Predict_scene('/home/pi/PRauto/FTPR6.h5',img)
    scene_nr=sc.scene_predict()
    pp=Scene_postprocess(img,scene_nr)
    img=pp.postprocess()
    img_denoise=Denoise_img(img,"/home/pi/PRauto/DNGEN116.h5", 0.7)
    denoised=img_denoise.histogram_norm()
    denoised=img_denoise.denoise()
    srgan_model=Upscale_img(denoised,"/home/pi/PRauto/GEN146v10-2x.h5", "/home/pi/PRauto/GEN144v3-4x.h5")
    upscaled=srgan_model.srgan_pipeline()
    save=Save_file(upscaled,input_img,output_path)
    save.save_pipeline()
    e=time.perf_counter()
    print("Czas: ",e-s)

    gc.collect()
    K.clear_session()
    os.remove(img_path)

    #Ponowne uruchomienie programu
    os.system('python3 /home/pi/PRauto/PRauto3.py')
    
if __name__=='__main__':
    main()
