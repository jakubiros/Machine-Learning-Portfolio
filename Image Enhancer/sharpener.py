'''
Mały program do poprawiania jakości filmu w postaci wyostrzania i możliwości skalowania dwukrotnego.
Wyostrzanie i interpolacja odbywa się za pomocą wytrenowanych sieci konwolucyjnych na własnym zbiorze danych.
Sieci mają architekture na wzór ResNet i U-Net trenowane metodą bezwarunkowego GAN.
'''

import sys
import os
import time
import cv2
import numpy as np
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model,Model

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Funkcja odpowiadająca za zapisywanie filmu
def save_settings(input_path,output_path,codec='mp4v',fps=30,resolution=(1920,1080)):
    global cap
    cap=cv2.VideoCapture(input_path)
    fourcc=cv2.VideoWriter_fourcc(*codec)
    out=cv2.VideoWriter(output_path,fourcc,fps,resolution,isColor=True)
    return out

#Funkcja wczytująca sieć do pamięci
def model_load(model_name):
    model=load_model(model_name+'.h5')
    return model

#Funkcja sprawdzająca ilość kaltek w filmie
def read_all_frames():
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count
    return duration

#Funkcja pozwalająca dostosować wypełenienie wyostrzania
def layer_thresh(frame1,frame2,alpha):
    beta=(1.0-alpha)
    result=cv2.addWeighted(frame1,alpha,frame2,beta,0.0)
    return result

#Funkcja mierząca szacowany czas do ukończenia przetwarzania filmu
def measure_time(counter,num_frames):
    global start,end
    if counter==0:
        start=time.time()
    if counter==10:
        end=time.time()
        time_elapsed=end-start
        time_elapsed=round(time_elapsed,2)
        pred_time=((num_frames/10)*time_elapsed)/60
        pred_time=round(pred_time,2)
        return pred_time 

#Funkcja pozwalająca na wpisanie potrzebnych parametrów    
def config(scale=False):
    input_path=input('Ścieżka do pliku: ')
    output_path=input('Ścieżka zapisu pliku: ')

    codec=input('Kodek(mp4v,avc1,avc3,hev1,hvc1,DIVX,XVID): ')
    if not codec=='mp4v' or codec=='avc1' or codec=='avc3' or codec=='hev1' or codec=='hvc1' or codec=='DIVX' or codec=='XVID':
        print('Źle wpisałeś.')
        codec=input('Kodek(mp4v,avc1,avc3,hev1,hvc1,DIVX,XVID): ')
    else:
        pass
    fps=int(input('Ilość fps: '))
    if not type(fps)==int:
        fps=int(input('Ilość fps: '))  #zrobić przez try
    else:
        pass
    x=int(input('Wartość x rozdzielczości: '))
    y=int(input('Wartość y rozdzielczości: '))
    if scale==True:
        x=x*2
        y=y*2
    model_name=input('Model AI: ')
    threshold=float(input('Pokrycie %: '))
    threshold /=100
    resolution=(x,y)
    return input_path,output_path,codec,fps,threshold,resolution,model_name


#Funkcja odpowiadająca za wyostrzanie filmu za pomocą sieci 
def frame_cap_sharpen(resolution,thr):
    try:
        count=0
        while(cap.isOpened()):
            ret,frame=cap.read()
            all_frames=read_all_frames()
            if ret==True:
                frame=frame.astype('float32')/127.5-1
                frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame=cv2.resize(frame,resolution,interpolation=cv2.INTER_CUBIC)
                frame=np.expand_dims(frame,axis=0)
                score=model.predict(frame)
                score=score[0,:,:,:]
                frame=frame[0,:,:,:]
                score=layer_thresh(score,frame,thr)
                score=cv2.cvtColor(score,cv2.COLOR_BGR2RGB)
                score=((score+1)*127.5).astype('uint8')
                save_out.write(score)
                time_to_do=measure_time(count,all_frames)
                count +=1
                print('Klatki: ',str(count),'/',str(all_frames))
                if count==11:
                    print('-'*25)
                    print('Pozostały czas: ',time_to_do,'min.')
                    print('-'*25)
                live_score=cv2.resize(score,(720,480))
                cv2.imshow('frame',live_score)

                if cv2.waitKey(5)==ord('q'):
                    break
            else:
                break
    except KeyboardInterrupt:
        save_out.write(score)

#Funkcja która interrpoluje obraz 2-krotnie
def frame_cap_scale(thr,resolution):
    try:
        count=0
        while(cap.isOpened()):
            ret,frame=cap.read()
            all_frames=read_all_frames()
            if ret==True:
                frame=frame.astype('float32')/127.5-1
                frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame=np.expand_dims(frame,axis=0)
                score=model.predict(frame)
                score=score[0,:,:,:]
                frame=frame[0,:,:,:]
                frame=cv2.resize(frame,resolution,interpolation=cv2.INTER_CUBIC)
                score=layer_thresh(score,frame,thr)
                score=cv2.cvtColor(score,cv2.COLOR_BGR2RGB)
                score=((score+1)*127.5).astype('uint8')
                save_out.write(score)
                time_to_do=measure_time(count,all_frames)
                count +=1
                print('Klatki: ',str(count),'/',str(all_frames))
                if count==11:
                    print('-'*25)
                    print('Pozostały czas: ',time_to_do,'min.')
                    print('-'*25)
                live_score=cv2.resize(score,(720,480))
                cv2.imshow('frame',live_score)

                if cv2.waitKey(5)==ord('q'):
                    break
            else:
                break
    except KeyboardInterrupt:
        save_out.write(score)

print('1: Sharpener')
print('2: UpScaling 2x')

choice=int(input('Opcja: '))

if choice==1:
    print('Wybrałeś sharpenera')
    input_path,output_path,codec,fps,threshold,resolution,model_name=config(scale=False)
    save_out=save_settings(input_path,output_path,codec,fps,resolution)
    path=os.getcwd()
    model_path=path+'/models/'+model_name
    model=model_load(model_path)
    frame_cap_sharpen(resolution,threshold)
else:
    print('Wybrałeś upscaling')
    input_path,output_path,codec,fps,threshold,resolution,model_name=config(scale=True)
    save_out=save_settings(input_path,output_path,codec,fps,resolution)
    path=os.getcwd()
    model_path=path+'/models/'+model_name
    model=model_load(model_path)
    frame_cap_scale(threshold,resolution)
    

