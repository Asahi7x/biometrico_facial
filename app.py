"""
Mini proyecto: Biométrico facial con interfaz gráfica (PyQt6 + OpenCV)

Funciones:
  1. Iniciar / detener la cámara y ver el video en vivo con el rostro marcado.
  2. Registrar un usuario nuevo (captura 30 fotos de su rostro).
  3. Entrenar el modelo de reconocimiento (LBPH de OpenCV).
  4. Verificar identidad y marcar asistencia (se guarda en registros.csv).
  5. Ver la tabla de registros.

Instalar dependencias:
  pip install PyQt6 opencv-contrib-python numpy
Ejecutar:
  python app.py
"""

import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QMainWindow, QMessageBox, QPushButton, QStatusBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------- rutas
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")          # dataset/<nombre>/0.png ...
MODELO = os.path.join(BASE, "modelo.yml")        # modelo LBPH entrenado
ETIQUETAS = os.path.join(BASE, "etiquetas.json") # id numérico -> nombre
REGISTROS = os.path.join(BASE, "registros.csv")  # historial de asistencia

FOTOS_POR_USUARIO = 30
UMBRAL_CONFIANZA = 70  # LBPH: menor = más parecido. Ajustar si falla mucho.

ESTILO = """
QMainWindow, QDialog { background: #1e2230; }
QLabel { color: #e8eaf0; font-size: 14px; }
QLabel#titulo { font-size: 22px; font-weight: bold; color: #ffffff; }
QLabel#video { background: #000000; border: 2px solid #3a4160; border-radius: 8px; }
QLabel#resultado { font-size: 18px; font-weight: bold; padding: 8px;
                   border-radius: 6px; background: #2a3045; }
QPushButton { background: #3d5afe; color: white; border: none; padding: 10px;
              border-radius: 6px; font-size: 14px; }
QPushButton:hover { background: #536dfe; }
QPushButton:disabled { background: #444a60; color: #888; }
QPushButton#peligro { background: #c62828; }
QPushButton#exito { background: #2e7d32; }
QStatusBar { color: #aab; }
QTableWidget { background: #ffffff; font-size: 13px; }
"""


class VentanaRegistros(QDialog):
    """Pantalla secundaria: muestra el historial de asistencia."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Registros de asistencia")
        self.resize(520, 400)
        tabla = QTableWidget(0, 3)
        tabla.setHorizontalHeaderLabels(["Nombre", "Fecha", "Hora"])
        tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        if os.path.exists(REGISTROS):
            with open(REGISTROS, newline="", encoding="utf-8") as f:
                for fila in csv.reader(f):
                    r = tabla.rowCount()
                    tabla.insertRow(r)
                    for c, valor in enumerate(fila):
                        tabla.setItem(r, c, QTableWidgetItem(valor))
        cerrar = QPushButton("Cerrar")
        cerrar.clicked.connect(self.accept)
        lay = QVBoxLayout(self)
        lay.addWidget(tabla)
        lay.addWidget(cerrar)


class BiometricoFacial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biométrico Facial")
        self.resize(980, 600)

        # --- visión por computadora
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.reconocedor = None
        self.etiquetas = {}
        self.cargar_modelo()

        self.camara = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.actualizar_frame)

        # estado: "libre", "registrando", "verificando"
        self.modo = "libre"
        self.nombre_registro = ""
        self.fotos_tomadas = 0

        self.construir_interfaz()

    # ------------------------------------------------------------ interfaz
    def construir_interfaz(self):
        titulo = QLabel("Sistema Biométrico Facial")
        titulo.setObjectName("titulo")

        self.video = QLabel("Cámara apagada")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 480)

        self.btn_camara = QPushButton("▶  Iniciar cámara")
        self.btn_registrar = QPushButton("👤  Registrar usuario")
        self.btn_entrenar = QPushButton("🧠  Entrenar modelo")
        self.btn_verificar = QPushButton("✔  Verificar / Marcar asistencia")
        self.btn_verificar.setObjectName("exito")
        self.btn_registros = QPushButton("📋  Ver registros")
        self.btn_salir = QPushButton("✖  Salir")
        self.btn_salir.setObjectName("peligro")

        self.btn_camara.clicked.connect(self.alternar_camara)
        self.btn_registrar.clicked.connect(self.iniciar_registro)
        self.btn_entrenar.clicked.connect(self.entrenar)
        self.btn_verificar.clicked.connect(self.iniciar_verificacion)
        self.btn_registros.clicked.connect(lambda: VentanaRegistros(self).exec())
        self.btn_salir.clicked.connect(self.close)

        self.resultado = QLabel("Esperando…")
        self.resultado.setObjectName("resultado")
        self.resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resultado.setWordWrap(True)

        panel = QVBoxLayout()
        for b in (self.btn_camara, self.btn_registrar, self.btn_entrenar,
                  self.btn_verificar, self.btn_registros):
            panel.addWidget(b)
        panel.addSpacing(20)
        panel.addWidget(QLabel("Resultado:"))
        panel.addWidget(self.resultado)
        panel.addStretch()
        panel.addWidget(self.btn_salir)

        cuerpo = QHBoxLayout()
        cuerpo.addWidget(self.video, stretch=3)
        cuerpo.addLayout(panel, stretch=1)

        raiz = QVBoxLayout()
        raiz.addWidget(titulo)
        raiz.addLayout(cuerpo)
        cont = QWidget()
        cont.setLayout(raiz)
        self.setCentralWidget(cont)

        self.setStatusBar(QStatusBar())
        self.actualizar_botones()

    def actualizar_botones(self):
        encendida = self.camara is not None
        self.btn_camara.setText("■  Detener cámara" if encendida else "▶  Iniciar cámara")
        self.btn_registrar.setEnabled(encendida and self.modo == "libre")
        self.btn_verificar.setEnabled(encendida and self.modo == "libre"
                                      and self.reconocedor is not None)
        n = len(self.etiquetas)
        self.statusBar().showMessage(
            f"Usuarios en el modelo: {n}   |   Modo: {self.modo}")

    # ------------------------------------------------------------ cámara
    def alternar_camara(self):
        if self.camara is None:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                QMessageBox.critical(self, "Error", "No se pudo abrir la cámara.")
                return
            self.camara = cam
            self.timer.start(30)  # ~33 fps
        else:
            self.timer.stop()
            self.camara.release()
            self.camara = None
            self.modo = "libre"
            self.video.setText("Cámara apagada")
        self.actualizar_botones()

    def actualizar_frame(self):
        ok, frame = self.camara.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)  # efecto espejo
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = self.detector.detectMultiScale(gris, 1.2, 5, minSize=(100, 100))

        for (x, y, w, h) in rostros:
            rostro = cv2.resize(gris[y:y + h, x:x + w], (200, 200))
            color = (255, 200, 0)

            if self.modo == "registrando":
                carpeta = os.path.join(DATASET, self.nombre_registro)
                cv2.imwrite(os.path.join(carpeta, f"{self.fotos_tomadas}.png"), rostro)
                self.fotos_tomadas += 1
                self.resultado.setText(
                    f"Capturando {self.fotos_tomadas}/{FOTOS_POR_USUARIO}")
                if self.fotos_tomadas >= FOTOS_POR_USUARIO:
                    self.modo = "libre"
                    self.resultado.setText(
                        f"{self.nombre_registro} registrado.\nAhora presiona «Entrenar».")
                    self.actualizar_botones()

            elif self.modo == "verificando":
                id_, distancia = self.reconocedor.predict(rostro)
                if distancia < UMBRAL_CONFIANZA:
                    nombre = self.etiquetas.get(str(id_), "?")
                    color = (0, 200, 0)
                    self.marcar_asistencia(nombre)
                    self.resultado.setText(f"✔ Acceso concedido\n{nombre}")
                    self.resultado.setStyleSheet("background:#2e7d32;")
                else:
                    color = (0, 0, 255)
                    self.resultado.setText("✖ Rostro no reconocido")
                    self.resultado.setStyleSheet("background:#c62828;")
                self.modo = "libre"
                QTimer.singleShot(3000, lambda: self.resultado.setStyleSheet(""))
                self.actualizar_botones()

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            break  # solo procesamos el primer rostro

        self.mostrar(frame)

    def mostrar(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio))

    # ------------------------------------------------------------ acciones
    def iniciar_registro(self):
        nombre, ok = QInputDialog.getText(self, "Registrar usuario", "Nombre completo:")
        nombre = nombre.strip()
        if not ok or not nombre:
            return
        os.makedirs(os.path.join(DATASET, nombre), exist_ok=True)
        self.nombre_registro = nombre
        self.fotos_tomadas = 0
        self.modo = "registrando"
        self.resultado.setText("Mira a la cámara…")
        self.actualizar_botones()

    def entrenar(self):
        if not hasattr(cv2, "face"):
            QMessageBox.critical(self, "Falta módulo",
                                 "Instala opencv-contrib-python para usar LBPH.")
            return
        if not os.path.isdir(DATASET) or not os.listdir(DATASET):
            QMessageBox.warning(self, "Sin datos", "Primero registra al menos un usuario.")
            return
        imagenes, ids, etiquetas = [], [], {}
        for i, nombre in enumerate(sorted(os.listdir(DATASET))):
            carpeta = os.path.join(DATASET, nombre)
            if not os.path.isdir(carpeta):
                continue
            etiquetas[str(i)] = nombre
            for archivo in os.listdir(carpeta):
                img = cv2.imread(os.path.join(carpeta, archivo), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imagenes.append(img)
                    ids.append(i)
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.train(imagenes, np.array(ids))
        rec.write(MODELO)
        with open(ETIQUETAS, "w", encoding="utf-8") as f:
            json.dump(etiquetas, f, ensure_ascii=False, indent=2)
        self.reconocedor, self.etiquetas = rec, etiquetas
        self.resultado.setText(f"Modelo entrenado con {len(etiquetas)} usuario(s).")
        self.actualizar_botones()

    def iniciar_verificacion(self):
        self.modo = "verificando"
        self.resultado.setText("Verificando…")
        self.actualizar_botones()

    def marcar_asistencia(self, nombre):
        ahora = datetime.now()
        with open(REGISTROS, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [nombre, ahora.strftime("%Y-%m-%d"), ahora.strftime("%H:%M:%S")])

    def cargar_modelo(self):
        if hasattr(cv2, "face") and os.path.exists(MODELO) and os.path.exists(ETIQUETAS):
            self.reconocedor = cv2.face.LBPHFaceRecognizer_create()
            self.reconocedor.read(MODELO)
            with open(ETIQUETAS, encoding="utf-8") as f:
                self.etiquetas = json.load(f)

    def closeEvent(self, evento):
        if self.camara is not None:
            self.camara.release()
        evento.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(ESTILO)
    ventana = BiometricoFacial()
    ventana.show()
    sys.exit(app.exec())
