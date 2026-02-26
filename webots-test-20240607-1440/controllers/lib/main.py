#!/usr/bin/env python

#INIT de la GPIO AU TOUT DEBUT pour eviter les courts circuits
import mygpio
mygpio.initGPIO()

import mylidar
import mymotor

import myalgo
import mybuz

import mycom

from myalgo import *
from mybuz import *
from mycom import *
from mygpio import *
from mylidar import *
from mymotor import *

#INIT pour la conduite
is_crashed = False
crashed_timer = 0
chosen_opening={'dist':0, 'angle':0, 'aperture':0, 'width':0, 'length':0, 'dir':0}
angle_degre = 0
time_highturn = 0

#CONF pour la conduite
seuil_vitesse_max = 10

###########################################################
try :
#commencement du code principal
    #boucle infinie
    while True:
        
        #recuperation des donner du lidar
        tableau_lidar_mm = mylidar.lidar.read_lidar_measurements()
        
        #code de test pour jouer un son avec le buzer
        '''if angle_degre <-5:
            play_melody(melody1)
        elif (angle_degre <5) and (angle_degre >-5):
            play_melody(melody2)
        elif angle_degre > 5:
            play_melody(melody3)'''
        
        #calcule et condition pour la partie de l'inteligence de la voiture
        if (is_crashed == True):
            if (crashed_timer < 40):
                if (crashed_timer == 0):
                    servo(-angle_degre)
                    envoie_donner( [10],[0])
                    crashed_timer += 1
                else:
                    is_crashed = False
                    servo(angle_degre)
                    crashed_timer = 0
        else:
            min_dists = get_min_dists(tableau_lidar_mm)
            if (min_dists['forward'] < 200):
                #print("min_dists['forward']:", min_dists['forward'])
                is_crashed = True
                continue
            valid_openings = get_valid_openings(tableau_lidar_mm)
            chosen_opening = choose_opening(chosen_opening, valid_openings)       
            angle_degre=chosen_opening['angle']
            if abs(chosen_opening['angle'] < 17 or chosen_opening['dist'] < 400):
                angle_degre+=chosen_opening['dir']*15
            if (abs(chosen_opening['angle']) <= 28):
                if (chosen_opening['dir'] == 1):
                    angle_degre = (angle_degre*(min_dists['left'])+1*(min_dists['left']-min_dists['right'])*(10000/(min_dists['left']+1)))/(min_dists['left']+10000/(min_dists['left']+1))
                else:
                    angle_degre = (angle_degre*(min_dists['right'])+1*(min_dists['left']-min_dists['right'])*(10000/(min_dists['right']+1)))/(min_dists['right']+10000/(min_dists['right']+1))

            angle_degre *= 800/(min_dists['forward']+1)
            if (abs(chosen_opening['angle'])>28):
                angle_degre = chosen_opening['angle']

            angle_degre = min([200,angle_degre])
            

            dist = min_dists['forward']
            vitesse_m_s = (500+dist)/2000
            if (abs(angle_degre) < 40 or abs(chosen_opening['angle'])<40):
                time_highturn = 0
            else:
                time_highturn += 1
                if (time_highturn > 100):
                    vitesse_m_s *= 1.5
                    
            #fonction pour le controle
            servo(-max([-200,angle_degre]))
            vitesse_final = vitesse_m_s/0.11
            if(vitesse_final >=seuil_vitesse_max):
               vitesse_final =  seuil_vitesse_max
            print("vitesse final:",(int)(vitesse_final))
            #valeur_telemetre, valeur_accelerometre = execution_transmission([(int)(vitesse_final)], [10])
            envoie_donner([10],[(int)(vitesse_final)])
            
except KeyboardInterrupt :
    mycom.envoie_donner([0],[0])
    mymotor.destroy_servo()
    mycom.spi.close()
    mylidar.lidar.disconnect_lidar()
