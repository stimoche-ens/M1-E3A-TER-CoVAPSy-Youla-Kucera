#!/usr/bin/env python

import mytoggle
import numpy as np
import mymotor

#INIT pour la conduite
chosen_opening = {'dist':0, 'angle':0, 'aperture':0, 'width':0, 'length':0, 'dir':0}
highturn_timer = 0
mode_backward_timer = 40

def calcule_ouverture(x1,x2):
    if x1 > x2:
        a = x2 * np.sin(0.225)
        b = x1 - x2 * np.cos(0.225)
    else:
        a = x1 * np.sin(0.225)
        b = x2 - x1 * np.cos(0.225)
    
    return np.sqrt(a*a + b*b)
    
def get_min_dists(tableau_lidar_mm):
    #min_forward_dist = tableau_lidar_mm[0]
    forward_list = [j for i in [tableau_lidar_mm[-5:],tableau_lidar_mm[:5]] for j in i]
    min_forward_dist = min(1000 if i == 0 else i for i in forward_list)
    min_left_dist = min(1000 if i == 0 else i for i in tableau_lidar_mm[5:60])
    min_right_dist = min(1000 if i == 0 else i for i in tableau_lidar_mm[-60:-5])
    min_dist = min([min_forward_dist,min_left_dist,min_right_dist])
    return {'forward': min_forward_dist, 'left': min_left_dist, 'right': min_right_dist, 'min': min_dist}


def choose_opening(valid_openings):
    maxlength = max(d['length'] for d in valid_openings)
    return [d for d in valid_openings if d['length'] == maxlength][0]


def get_valid_openings(tableau_lidar_mm):
    valid_openings=[]
    i=-90
    while (i < 91):
        if (calcule_ouverture(tableau_lidar_mm[i+1],tableau_lidar_mm[i]) > 450):
            opening_dist = tableau_lidar_mm[i]
            opening_length = tableau_lidar_mm[i+1] - opening_dist
            opening_angle=i
            opening_aperture=0
            while (i < 91):
                i+=1
                if (calcule_ouverture(tableau_lidar_mm[i], opening_dist) > 100):
                    opening_aperture+=1
                else:
                    break
            opening_width=opening_dist*2*np.sin(np.pi*opening_aperture/180/2)
            if (abs(opening_width)>200):
                valid_openings.append({'dist':opening_dist, 'length':opening_length, 'width':opening_width, 'angle':opening_angle, 'aperture':opening_aperture, 'dir':1})
        i+=1
    i=90
    while (i > -91):
        if (calcule_ouverture(tableau_lidar_mm[i-1],tableau_lidar_mm[i]) > 450):
            opening_dist = tableau_lidar_mm[i]
            opening_length = tableau_lidar_mm[i-1] - opening_dist
            opening_angle=i
            opening_aperture=0
            while (i > -91):
                i-=1
                if (calcule_ouverture(tableau_lidar_mm[i],opening_dist) > 100):
                    opening_aperture+=1
                else:
                    break
            opening_width=opening_dist*2*np.sin(np.pi*opening_aperture/180/2)
            if (abs(opening_width)>200):
                valid_openings.append({'dist':opening_dist, 'length':opening_length, 'width':opening_width, 'angle':opening_angle, 'aperture':opening_aperture, 'dir':-1})
        i-=1
    return valid_openings


def get_angle_for_distancekeeping(min_dists):
    print("distancekeeping")
    return 0.04*(min_dists['left']-min_dists['right'])

def get_angle_for_opening(min_dists):
    wheeldir_deg = chosen_opening['angle']
    decal = (180/np.pi)*np.arctan(200/chosen_opening['dist'])
    wheeldir_deg = wheeldir_deg + chosen_opening['dir']*decal
    return wheeldir_deg

def get_angle(min_dists):
    if (chosen_opening['dir'] == 0):
        return get_angle_for_distancekeeping(min_dists)
    return get_angle_for_distancekeeping(min_dists)

def get_speed(min_dists, wheeldir_deg):
    global highturn_timer

    speed_m_s = (1500+min_dists['forward'])/3600
    print("min_dists['forward']:",min_dists['forward'])
    if (abs(wheeldir_deg) < 40 or abs(chosen_opening['angle'])<40):
        highturn_timer = 0
    else:
        highturn_timer += 1
        if (highturn_timer > 100):
            speed_m_s *= 1.5
    return speed_m_s

def mode_backward(min_dists):
    global mode_backward_timer

    mode_backward_timer
    wheeldir_deg=get_angle(min_dists)
    mymotor.set_wheeldir_deg[mytoggle.SEL_FN_INDEX](-wheeldir_deg)
    mymotor.set_recule[mytoggle.SEL_FN_INDEX]()
    #mymotor.set_final_speed(speed_m_s)
    mode_backward_timer+=1


def mode_forward(chosen_opening,min_dists):
    wheeldir_deg = get_angle(min_dists)
    if abs(chosen_opening['angle']) > 50:
        decal_distobst = 10-chosen_opening['dist']*np.cos(chosen_opening['angle']*np.pi/180)/100
        wheeldir_deg = wheeldir_deg - chosen_opening['dir']*decal_distobst
    #speed_m_s = max(1,get_speed(min_dists, wheeldir_deg))
    speed_m_s = min(5, get_speed(min_dists, wheeldir_deg))
    mymotor.set_wheeldir_deg[mytoggle.SEL_FN_INDEX](wheeldir_deg)
    mymotor.set_speed_m_s[mytoggle.SEL_FN_INDEX](speed_m_s)
    #mymotor.set_final_speed(speed_m_s)
    #valeur_telemetre, valeur_accelerometre = execution_transmission([(int)(vitesse_final)], [10])

def main_algo(tableau_lidar_mm):
    global chosen_opening
    global mode_backward_timer
    

    min_dists = get_min_dists(tableau_lidar_mm)
    valid_openings = get_valid_openings(tableau_lidar_mm)
    if len(valid_openings) > 0:
        chosen_opening = choose_opening(valid_openings)
        print("chosen_opening['angle']:",chosen_opening['angle'])
    if (mode_backward_timer < 40):
        mode_backward(min_dists)
    else:
        if (min_dists['forward'] < 200):
            mode_backward_timer = 0
            mode_backward(min_dists)
        else:
            mode_forward(chosen_opening,min_dists)

