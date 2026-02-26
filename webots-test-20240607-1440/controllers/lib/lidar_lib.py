from rplidar import RPLidar, RPLidarException
import time
"""
pour utiliser cette librairie il est neccessaire d'intaller les bibliothèque rplidar (pip install rplidar)
et rplidar-roboticia (pip install rplidar-roboticia)
"""
class LidarLibrary:
    def __init__(self, port='/dev/ttyUSB0', baudrate=256000, motor_speed=800):#vitesse motor max 1023
        """
        Initialise une instance de la classe LidarLibrary avec les paramètres de communication.
        
        Args:
            port (str): Le port série auquel le Lidar est connecté. Par défaut, '/dev/ttyUSB0'.
            baudrate (int): Le débit en bauds pour la communication série. Par défaut, 256000.
            motor_speed (int): La vitesse du moteur du Lidar. Par défaut, 1023.
        """
        self.PORT = port
        self.BAUDRATE = baudrate
        self.VITESSE_MOTEUR = motor_speed
        self.tab_angle = [0] * 361  # Tableau pour stocker les mesures à chaque angle
        self.lidar = None  # Initialisation de l'objet Lidar à None

    def connect_lidar(self):
        """
        Établit une connexion avec le Lidar.
        
        Tente de se connecter au Lidar en utilisant les paramètres spécifiés dans le constructeur.
        Affiche un message en cas de succès ou une erreur en cas d'échec.
        """
        try:
            self.lidar = RPLidar(self.PORT, baudrate=self.BAUDRATE)  # Initialisation du Lidar
            self.init_lidar()  # Appel de la méthode pour initialiser le Lidar
            print('Connexion avec le Lidar établie')
        except Exception as e:
            print("Erreur lors de la connexion au Lidar:", e)
            self.connect_lidar()
            self.read_lidar_measurements()

    def init_lidar(self):
        """
        Initialise les paramètres du Lidar.
        
        Connexion au Lidar, réinitialisation, configuration de la vitesse du moteur et démarrage des acquisitions.
        """
        self.lidar.connect()  # Connexion au Lidar
        self.lidar.reset()  # Réinitialisation du Lidar
        self.lidar.motor_speed = self.VITESSE_MOTEUR  # Configuration de la vitesse du moteur
        self.lidar.start()  # Démarrage des acquisitions
        time.sleep(0.5)  # Attente de 0.5 seconde pour la stabilisation

    def read_lidar_measurements(self, max_point_mes = 2000):
        """
        Lit les mesures du Lidar.
        
        Parcourt les scans du Lidar, met à jour le tableau des mesures avec les distances mesurées à chaque angle.
        En cas d'erreur de lecture, arrête le Lidar et affiche l'erreur.
        """
        try:
            for scan in self.lidar.iter_scans(max_buf_meas = max_point_mes):  # Boucle pour parcourir les scans
                for (_, angle, distance) in scan:  # Boucle pour extraire les angles et distances
                    if (270< angle < 360) or (0 < angle <90)  :
                        self.tab_angle[int(angle)] = distance  # Mise à jour du tableau des mesures
                return self.tab_angle  # Retourne le tableau des mesures après la lecture d'un scan
        except RPLidarException as e:
            print("Erreur lors de la lecture du Lidar:", e)
            self.lidar.stop()  # Arrête les acquisitions
            self.lidar.reset()  # Réinitialise le Lidar
            self.lidar.stop_motor() # Arrête le moteur du Lidar
            self.connect_lidar()
            self.read_lidar_measurements()
            print("reconecte")

    def disconnect_lidar(self):
        """
        Déconnecte proprement le Lidar.
        
        Arrête les acquisitions, arrête le moteur du Lidar et déconnecte le Lidar.
        """
        if self.lidar:
            self.lidar.stop()  # Arrête les acquisitions
            self.lidar.stop_motor()  # Arrête le moteur du Lidar
            self.lidar.disconnect()  # Déconnexion du Lidar
            print("Connexion avec le Lidar fermée")
        else:
            print("Le Lidar n'est pas connecté.")

if __name__ == "__main__":
    lidar_lib = LidarLibrary()  # Instanciation de la classe LidarLibrary
    lidar_lib.connect_lidar()  # Connexion au Lidar
    try:
        while True:
            tab_angle = lidar_lib.read_lidar_measurements(max_point_mes = 5000)  # Lecture des mesures du Lidar
            # Ici vous pouvez faire ce que vous voulez avec les données du Lidar
            print(tab_angle[180])  # Impression de la mesure à l'angle 90
            #time.sleep(0.1)  # Attente d'une seconde avant la prochaine lecture
    except KeyboardInterrupt:
        lidar_lib.disconnect_lidar()  # Déconnexion propre du Lidar en cas d'interruption

