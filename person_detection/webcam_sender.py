#!/usr/bin/env python3
"""
webcam_sender.py — Capture la webcam du PC et envoie les frames
à l'ESP32 via le port série pour la détection de personnes.

Usage:
    python webcam_sender.py --port COM5
    python webcam_sender.py --port COM5 --baud 115200 --camera 0

Dépendances:
    pip install opencv-python pyserial
"""

import argparse
import sys
import time

import cv2
import serial


IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT  # 9216 bytes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Envoie les images de la webcam à l'ESP32 pour détection de personnes"
    )
    parser.add_argument(
        "--port", type=str, required=True,
        help="Port série de l'ESP32 (ex: COM5, /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baudrate du port série (défaut: 115200)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Index de la webcam à utiliser (défaut: 0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  WEBCAM → ESP32 : Détection de Personnes")
    print("=" * 60)
    print()

    # Initialisation de la connexion série
    try:
        # timeout=1.0 permet de ne pas bloquer indéfiniment sur readline()
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
        time.sleep(2)  # Attendre que l'ESP32 se stabilise après reset DTR
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"[OK] Port série ouvert : {args.port} @ {args.baud} baud")
    except serial.SerialException as e:
        print(f"[ERREUR] Impossible d'ouvrir {args.port}: {e}")
        sys.exit(1)

    # Initialisation de la webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la caméra {args.camera}")
        sys.exit(1)
    print(f"[OK] Webcam {args.camera} ouverte")

    print("[INFO] Envoi de la commande 'detect_serial' à l'ESP32...")
    ser.write(b"\r")
    time.sleep(0.1)
    ser.write(b"detect_serial\r")
    ser.flush()

    print("[INFO] Démarrage de la boucle de détection. Appuyez sur 'q' pour quitter.\n")

    frame_count = 0
    last_score = None

    try:
        while True:
            # 1. Attendre READY
            ready_found = False
            start_wait = time.time()
            while time.time() - start_wait < 15:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue  # Timeout de 1s du readline
                
                print(f"  [ESP32] {line}")
                if "READY" in line:
                    ready_found = True
                    break
                    
            if not ready_found:
                print("[ERREUR] Timeout en attente du signal READY de l'ESP32")
                break

            # 2. Capturer et envoyer
            ret, frame = cap.read()
            if not ret:
                print("[ERREUR] Capture webcam échouée")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            
            raw_bytes = resized.tobytes()
            ser.write(raw_bytes)
            ser.flush()
            
            frame_count += 1
            print(f"  [PC] Frame #{frame_count} envoyée ({len(raw_bytes)} octets)")

            # 3. Attendre le résultat (scores + FRAME_OK)
            start_wait = time.time()
            frame_ok = False
            
            while time.time() - start_wait < 10:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                
                print(f"  [ESP32] {line}")
                
                if "person score:" in line.lower():
                    try:
                        parts = line.lower().split("person score:")
                        if len(parts) >= 2:
                            score_str = parts[1].split("%")[0].strip()
                            last_score = int(score_str)
                    except Exception:
                        pass
                elif "FRAME_OK" in line:
                    frame_ok = True
                    break
                    
            if not frame_ok:
                print("[ERREUR] Timeout en attente du résultat de l'ESP32")
                break

            # 4. Affichage OpenCV (UI Optionnelle)
            display = frame.copy()
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)

            if last_score is not None:
                if last_score >= 80:
                    color = (0, 0, 255)
                    label = f"PERSONNE DETECTEE ({last_score}%)"
                else:
                    color = (0, 255, 0)
                    label = f"Pas de personne ({last_score}%)"
                cv2.putText(display, label, (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display, f"Frame #{frame_count} - En attente...", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Preview réduite envoyée à l'ESP32
            preview_small = cv2.resize(resized, (192, 192), interpolation=cv2.INTER_NEAREST)
            preview_bgr = cv2.cvtColor(preview_small, cv2.COLOR_GRAY2BGR)
            cv2.putText(preview_bgr, "96x96 -> ESP32", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imshow("Webcam - Detection de Personnes", display)
            cv2.imshow("Image envoyee (96x96)", preview_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Arrêt demandé par l'utilisateur.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interruption clavier.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()
        print(f"\n[INFO] Terminé. {frame_count} frames envoyées.")


if __name__ == "__main__":
    main()
