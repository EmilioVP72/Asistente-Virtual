import math 
import threading
import cv2
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from queue import Queue

# ==========================
# Configuración de Mediapipe
# ==========================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cola para compartir datos entre hilos
pose_data_queue = Queue()

def calculate_angle(a, b, c):
    ab = [b.x - a.x, b.y - a.y, b.z - a.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]
    
    dot_product = sum([ab[i] * bc[i] for i in range(3)])
    mag_ab = math.sqrt(sum([ab[i] ** 2 for i in range(3)]))
    mag_bc = math.sqrt(sum([bc[i] ** 2 for i in range(3)]))
    
    angle = math.degrees(math.acos(dot_product / (mag_ab * mag_bc)))
    return angle

def evaluate_squat(knee_angle):
    if 80 <= knee_angle <= 100:
        return "Correcta"
    elif knee_angle < 80:
        return "Muy alta"
    else:
        return "Muy baja"

def mediapipe_pose_detection():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cadera_izq = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                rodilla_izq = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                tobillo_izq = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                cadera_der = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                rodilla_der = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                tobillo_der = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

                left_knee_angle = calculate_angle(cadera_izq, rodilla_izq, tobillo_izq)
                right_knee_angle = calculate_angle(cadera_der, rodilla_der, tobillo_der)
                squat_status = evaluate_squat(min(left_knee_angle, right_knee_angle))

                print(f"Ángulo rodilla izquierda: {left_knee_angle:.2f}°")
                print(f"Ángulo rodilla derecha: {right_knee_angle:.2f}°")
                print(f"Estado: {squat_status}")

                pose_data_queue.put(([left_knee_angle, right_knee_angle], squat_status))

            cv2.imshow('Detección de Sentadilla', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# Configuración de OpenGL
# ==========================
angle_left_knee = 0.0
angle_right_knee = 0.0
angle_left_hip = 0.0
angle_right_hip = 0.0
angle_back = 0.0
squat_status = "Evaluando..."

camera_mode = 0 

def init_opengl():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

def draw_cube():
    glutSolidCube(1)

def draw_sphere(color):
    glColor3fv(color)
    glutSolidSphere(0.2, 20, 20)

def draw_arm():
    glPushMatrix()
    glScalef(0.2, 1.0, 0.2)
    draw_cube()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.0, -1.0, 0.0)
    draw_sphere((0.8, 0.2, 0.2))
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.0, -1.0, 0.0)
    glScalef(0.2, 1.0, 0.2)
    draw_cube()
    glPopMatrix()

def draw_leg(angle_knee, angle_hip):
    glPushMatrix()
    glRotatef(angle_hip, 1, 0, 0)
    glScalef(0.2, 1.2, 0.2)
    draw_cube()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.0, -1.2, 0.0)
    draw_sphere((1.0, 0.0, 0.0))
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.0, -1.2, 0.0)
    glRotatef(angle_knee - 30, 1, 0, 0)
    glTranslatef(0.0, -1.2, 0.0)
    glScalef(0.2, 1.0, 0.2)
    draw_cube()
    glPopMatrix()

def draw_human(angle_left_knee, angle_right_knee, angle_left_hip, angle_right_hip, angle_back, squat_status):
    # Cabeza
    glPushMatrix()
    glColor3f(1.0, 0.8, 0.6)
    glTranslatef(0.0, 1.8, 0.0)
    glutSolidSphere(0.3, 20, 20)
    glPopMatrix()

    # Espalda con movimiento sincronizado
    glPushMatrix()
    glColor3f(0.5, 0.5, 0.5)
    glTranslatef(0.0, 0.8, 0.0)
    glRotatef(angle_back, 1, 0, 0)
    glScalef(0.5, 1.0, 0.3)
    draw_cube()
    glPopMatrix()

    # Pierna izquierda
    glPushMatrix()
    glTranslatef(-0.2, -0.5, 0.0)
    draw_leg(angle_left_knee, angle_left_hip)
    glPopMatrix()

    # Pierna derecha
    glPushMatrix()
    glTranslatef(0.2, -0.5, 0.0)
    draw_leg(angle_right_knee, angle_right_hip)
    glPopMatrix()

    # Brazo izquierdo extendido hacia adelante
    glPushMatrix()
    glColor3f(0.5, 0.5, 0.8)
    glTranslatef(-0.5, 1.0, -0.5)
    glRotatef(45, 1, 0, 0)
    draw_arm()
    glPopMatrix()

    # Brazo derecho extendido hacia adelante
    glPushMatrix()
    glColor3f(0.5, 0.5, 0.8)
    glTranslatef(0.5, 1.0, -0.5)
    glRotatef(45, 1, 0, 0)
    draw_arm()
    glPopMatrix()

    # Estado de la sentadilla (texto con colores dinámicos)
    glPushMatrix()
    if squat_status == "Correcta":
        glColor3f(0.0, 1.0, 0.0)  
    else:
        glColor3f(1.0, 0.0, 0.0)  
    glRasterPos3f(-1.5, 2.5, 0)
    for char in squat_status:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
    glPopMatrix()

def display():
    global angle_left_knee, angle_right_knee, angle_left_hip, angle_right_hip, angle_back, squat_status

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    if camera_mode == 0: 
        gluLookAt(0, 2.0, 10, 0, 1.0, 0, 0, 1, 0)
    else: 
        gluLookAt(8, 5, 8, 0, 1.0, 0, 0, 1, 0)

    draw_human(angle_left_knee, angle_right_knee, angle_left_hip, angle_right_hip, angle_back, squat_status)

    glutSwapBuffers()

    if not pose_data_queue.empty():
        angles, squat_status = pose_data_queue.get()
        angle_left_knee, angle_right_knee = angles
        angle_left_hip = 90 - min(angle_left_knee, angle_right_knee)
        angle_right_hip = angle_left_hip

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w / h, 1, 50)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    global camera_mode
    if key == b'c':
        camera_mode = (camera_mode + 1) % 2
    elif key == b'q':
        sys.exit()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("Simulador de Sentadilla")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutIdleFunc(display)
    glutKeyboardFunc(keyboard)
    init_opengl()

    mediapipe_thread = threading.Thread(target=mediapipe_pose_detection, daemon=True)
    mediapipe_thread.start()

    glutMainLoop()

if __name__ == "__main__":
    main()
