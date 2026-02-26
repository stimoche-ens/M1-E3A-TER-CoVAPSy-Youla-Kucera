#!/usr/bin/env python

import mytoggle


global ServoPWM
global MotorPWM
global ServoPin
global ServoFreq
global MotorPin
global MotorFreq

global BTN1_GPIO

def init_GPIO_SIM():
    print("not initializing GPIO because simulating.")

def init_GPIO_RPI():
    import RPi.GPIO as GPIO
    
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()

    global ServoPWM
    global MotorPWM
    global ServoPin
    global ServoFreq
    global MotorPin
    global MotorFreq
    global BTN1_GPIO
    ServoPin = 13
    ServoFreq =  50
    MotorPin = 12
    MotorFreq =  50

    #init servo et moteur
    GPIO.setup(ServoPin, GPIO.OUT)
    GPIO.output(ServoPin, GPIO.LOW)
    ServoPWM = GPIO.PWM(ServoPin, ServoFreq)
    GPIO.setup(MotorPin, GPIO.OUT)
    GPIO.output(MotorPin, GPIO.LOW)
    MotorPWM = GPIO.PWM(MotorPin, MotorFreq)

    ServoPWM.start(0)
    MotorPWM.start(0)

    BTN1_GPIO = 5
    GPIO.setup(BTN1_GPIO, GPIO.IN)

def destroy_GPIO_SIM():
    print("not destroying because simulating")

def destroy_GPIO_RPI():
    import RPi.GPIO as GPIO

    ServoPWM.stop()
    MotorPWM.stop()
    GPIO.cleanup()

init_GPIO = [0,0]
init_GPIO[mytoggle.SIM_FN_INDEX] = init_GPIO_SIM
init_GPIO[mytoggle.RPI_FN_INDEX] = init_GPIO_RPI


destroy_GPIO = [0,0]
destroy_GPIO[mytoggle.SIM_FN_INDEX] = destroy_GPIO_SIM
destroy_GPIO[mytoggle.RPI_FN_INDEX] = destroy_GPIO_RPI
