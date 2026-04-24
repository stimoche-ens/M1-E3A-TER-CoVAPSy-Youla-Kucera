import os
import rclpy
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32
from enum import Enum
import time
from joblib import load
from ament_index_python.packages import get_package_share_directory

# Definition des etats possibles pour le vehicule
class EtatVehicule(Enum):
    INFERENCES = 1
    DEGAGEMENT_Droite = 2
    DEGAGEMENT_Gauche = 3
    DEGAGEMENT_Avant = 4

class CoVAPSy_conduiteR(Node):
    def __init__(self):
        super().__init__('CoVAPSy_conduiteR')

       # Load model + scalers
        pkg_share_dir = get_package_share_directory('YKREN')
        #self.model = load("/home/voituremaxime/ros2_ws/src/YKREN/YKREN/trained_model.joblib")
        self.model =    load(os.path.join(pkg_share_dir, "models", "trained_model.joblib"))
        self.scaler_X = load(os.path.join(pkg_share_dir, "models", "scaler_X.joblib"))
        self.scaler_Y = load(os.path.join(pkg_share_dir, "models", "scaler_Y.joblib"))

    
        # Initialisation des variables pour la machine a etats
        self.etat_courant = EtatVehicule.INFERENCES
        self.seuil_distance_mur = 0.15 # Distance minimale en metres avant blocage
        self.temps_debut_recul = 0.0
        self.duree_recul_sec = 1.6 # Duree de la manoeuvre de recul
        self.vitesse_recul = -0.9 # Consigne de vitesse pour le recul (m/s)
        self.vitesse_avance = 0.8 # Consigne de vitesse pour l'avance (m/s)

        # Nouvelles variables pour éviter le bouclage
        self.temps_fin_degagement = 0.0
        self.delai_inhibition_sec = 2.0 # Temps d'attente (en secondes) avant d'autoriser une nouvelle détection de blocage

        self.vitesse_mesuree = 0.0
        self.compteur_vitesse_nulle = 0 # Initialisation du compteur de valeurs nulles consécutives
        self.distance_ultrason = 999.0

        # Initialisation des interfaces ROS
        self.__ackermann_publisher = self.create_publisher(AckermannDrive, 'cmd_ackermann', 1) # speed in m/s, steering angle in degrees
        self.create_subscription(LaserScan,'scan', self.__on_lidar_acquisition,1) # distance in meters
        self.create_subscription(Float32, 'vitesse_mesuree', self.__vitesse_mesuree_callback, 1)
        self.create_subscription(Float32, 'distance_ultrason', self.__ultrason_callback, 1)
        self.get_logger().info("[CoVAPSy_conduiteR] Modèle chargé et noeud prêt.")


    def __vitesse_mesuree_callback(self, message):
        self.vitesse_mesuree = message.data
        # Incrémentation ou réinitialisation du compteur
        if abs(self.vitesse_mesuree) == 0.0:
            self.compteur_vitesse_nulle += 1
        else:
            self.compteur_vitesse_nulle = 0

    def __ultrason_callback(self, message):
        # Mise à jour de la distance ultrasons lue depuis la série
        self.distance_ultrason = message.data

    def angle_to_index(self, angle_deg, scan_msg):
        # function converting angle in degrees to index 
        if angle_deg>180: #modulo 2pi to keep the angle between -180 and 180
            angle_deg+=360
        elif angle_deg<-180: #modulo 2pi to keep the angle between -180 and 180
            angle_deg-=360
        angle_rad = np.deg2rad(angle_deg)
        idx = int(angle_rad / scan_msg.angle_increment) # angles positive à gauche, angles négatifs à droite, angle 0° au milieu
        return idx

    def generate_median_scan(self, message, start_angle=-100, end_angle=100, step=1):
            # function taking the median of the points in each angle range
            angles = range(start_angle, end_angle + step, step)  # inclut end_angle
            median_scan = []
            LidarTable = list(message.ranges) 
            for i, angle in enumerate(angles): # go over each angle in the range we're interested in
                next_angle = angle + step
                start_idx = self.angle_to_index(angle,message)
                end_idx = self.angle_to_index(next_angle,message)
                self.get_logger().info(f'angle = {angle:.2f}°, start_idx = {start_idx}, end_idx = {end_idx}')

                tab=[]
                for j in range(min(start_idx, end_idx), max(start_idx, end_idx) + 1): # go over the indices corresponding to the angle range
                    tab.append(LidarTable[j])
                if tab: # if there are points in the range, calculate the median, otherwise append NaN
                    median_scan.append(np.median(tab))
                else:
                    median_scan.append(float('nan'))
            return median_scan

    def __on_lidar_acquisition(self, scan_msg):
         # Generate the fixed LiDAR table 
        LidarTable_fixed = self.generate_median_scan(scan_msg,-100, 100, 1) # 201 points from -100° to 100° 
        
        # Convert LiDAR to numpy
        lidarTable_fixed = np.array(LidarTable_fixed, dtype=np.float32)

        # Replace inf / nan
        lidar = np.nan_to_num(lidarTable_fixed, nan=30.0, posinf=30.0, neginf=0.0)
        current_lidar= np.nan_to_num(lidarTable_fixed, nan=0.0, posinf=0.0, neginf=0.0)

        # Preprocess (normalisation)
        lidar_scaled = self.scaler_X.transform([lidar])   # shape (1, N)
        current_lidar_normalise = current_lidar / 12

        # Filtrage des valeurs nulles pour evaluer la distance minimale reelle
        valeurs_valides = current_lidar[current_lidar > 0.0]
        
        if len(valeurs_valides) > 0:
            distance_minimale = np.min(valeurs_valides)
        else:
            distance_minimale = 12.0 

        # Logique de transition d'etat
        if self.etat_courant == EtatVehicule.INFERENCES:
            
            # Condition de passage en etat de degagement
            if distance_minimale < self.seuil_distance_mur and self.vitesse_mesuree ==0.0 and self.distance_ultrason > 10.0:
                if np.argmin(valeurs_valides)<len(valeurs_valides)/2:
                    self.etat_courant=EtatVehicule.DEGAGEMENT_Gauche
                    self.get_logger().info('Blocage detecte. Passage en etat DEGAGEMENT_Gauche.')
                else:

                    self.etat_courant = EtatVehicule.DEGAGEMENT_Droite
                    self.get_logger().info('Blocage detecte. Passage en etat DEGAGEMENT_Droite.')
                self.temps_debut_recul = time.time()
                self.compteur_vitesse_nulle = 0
                
                return
            elif self.compteur_vitesse_nulle >= 4 and distance_minimale > self.seuil_distance_mur and (time.time() - self.temps_fin_degagement) > self.delai_inhibition_sec:
                self.etat_courant = EtatVehicule.DEGAGEMENT_Avant
                self.compteur_vitesse_nulle = 0
                self.get_logger().info('Vitesse nulle detectee. Passage en etat DEGAGEMENT AVANT.')
                return
    
            # Predict
            pred_scaled = self.model.predict(lidar_scaled)
            pred = self.scaler_Y.inverse_transform(pred_scaled)[0] 

            steering = float(pred[0]) # predicted steering angle in degrees
            speed = float(pred[1]) # predicted speed in m/s

            # Safety clipping
            steering = max(min(steering, 18.0), -18.0) # steering angle in degrees
            speed = max(min(speed, 0.8), 0.0) #speed in m/s, between 0 and 2 m/s for safety

            # Publish command
            cmd = AckermannDrive()
            cmd.steering_angle = steering # angle in degrees
            cmd.speed = speed # speed in m/s

            self.__ackermann_publisher.publish(cmd)
            
            self.get_logger().info(f"[Inference] v={speed:.2f} m/s, dir={steering:.2f} deg")

        elif self.etat_courant == EtatVehicule.DEGAGEMENT_Gauche:
            
            temps_ecoule = time.time() - self.temps_debut_recul
            
            # Verification du chronometre de recul
            if temps_ecoule < self.duree_recul_sec:
                # Execution de la commande de recul
                cmd = AckermannDrive()
                cmd.speed = float(self.vitesse_recul)
                cmd.steering_angle = -17.0
                self.__ackermann_publisher.publish(cmd)
            else:
                # Fin du degagement et retour a l'etat autonome
                self.etat_courant = EtatVehicule.INFERENCES
                # Reinitialisation des variables pour la reprise de l'inference
                self.current_speed = 0.1
                self.current_angle = 0.0
                self.previous_lidar = current_lidar_normalise.copy()
                self.get_logger().info('Fin du recul. Passage en etat INFERENCES.')

        elif self.etat_courant == EtatVehicule.DEGAGEMENT_Droite:
            
            temps_ecoule = time.time() - self.temps_debut_recul
            
            # Verification du chronometre de recul
            if temps_ecoule < self.duree_recul_sec:
                # Execution de la commande de recul
                cmd = AckermannDrive()
                cmd.speed = float(self.vitesse_recul)
                cmd.steering_angle = 17.0
                self.__ackermann_publisher.publish(cmd)
            else:
                # Fin du degagement et retour a l'etat autonome
                self.etat_courant = EtatVehicule.INFERENCES
                # Reinitialisation des variables pour la reprise de l'inference
                self.current_speed = 0.1
                self.current_angle = 0.0
                self.previous_lidar = current_lidar_normalise.copy()
                self.get_logger().info('Fin du recul. Passage en etat INFERENCES.')

        elif self.etat_courant == EtatVehicule.DEGAGEMENT_Avant:
            
            temps_ecoule = time.time() - self.temps_debut_recul
            
            # Verification du chronometre de recul
            if temps_ecoule < self.duree_recul_sec:
                # Execution de la commande de recul
                cmd = AckermannDrive()
                cmd.speed = -float(self.vitesse_recul)
                cmd.steering_angle = 0.0
                self.__ackermann_publisher.publish(cmd)
            else:
                # Fin du degagement et retour a l'etat autonome
                self.etat_courant = EtatVehicule.INFERENCES
                # Reinitialisation des variables pour la reprise de l'inference
                self.current_speed = 0.1
                self.current_angle = 0.0
                self.previous_lidar = current_lidar_normalise.copy()
                
                # Enregistrement de l'heure de fin de manœuvre
                self.temps_fin_degagement = time.time() 
                
                self.get_logger().info('Fin du recul. Passage en etat INFERENCES.')


def main(args=None):
    rclpy.init(args=args)
    noeud = CoVAPSy_conduiteR()
    rclpy.spin(noeud)
    noeud.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
