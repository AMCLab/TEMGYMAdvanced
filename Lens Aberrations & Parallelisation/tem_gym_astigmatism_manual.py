#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:46:50 2022

@author: andy
"""
import sys
import cv2
import numpy as np
from temgym_astigmatism import TemGym
import matplotlib.pyplot as plt
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import PyQt5
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QGridLayout,
    QRadioButton,
    QGroupBox,
    QPushButton,
)

from PyQt5.QtGui import QIcon, QFont

import pyqtgraph.opengl as gl
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *

app = PyQt5.QtWidgets.QApplication(sys.argv)
win = QMainWindow()
win.show()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000, 500)
win.setWindowTitle("TemGym Manual Alignment")

#Create Docks
d1 = Dock("Detector", size=(512, 512))  ## give this dock the minimum possible size
d2 = Dock("GUI", size=(500, 300), closable=True)
d3 = Dock("3D", size=(500, 400))

#Add docks to area
area.addDock(d1, "right")  ## place d1 at right edge of dock area
area.addDock(d2, "bottom", d1)  ## place d2 at bottom edge of d1
area.addDock(d3, "left")  ## place d3 at left edge of dock area (it will fill the whole space since there are no other docks yet)

#Create Detector Graphics objects
w1 = pg.GraphicsLayoutWidget()
v1 = w1.addViewBox()
v1.setAspectLocked(1.0)
img = pg.ImageItem(border="b")
v1.addItem(img)
d1.addWidget(w1)

# Create GUI Sliders
groupBox = QGroupBox("Component Controls")

slider_upper_quadrupole = QSlider(QtCore.Qt.Orientation.Horizontal)
slider_upper_quadrupole .setTickPosition(QSlider.TickPosition.TicksBelow)
slider_upper_quadrupole .setMinimum(-100)
slider_upper_quadrupole .setMaximum(100)
slider_upper_quadrupole .setValue(0)
slider_upper_quadrupole .setTickPosition(QSlider.TicksBelow)
slider_upper_quadrupole .setTickInterval(100)

upper_quadrupole_label = QLabel('Upper Quadrupole Current (Amperes) = ' + "{:.2f}".format(slider_upper_quadrupole.value()))
upper_quadrupole_label.setMinimumWidth(80)

slider_lower_quadrupole  = QSlider(QtCore.Qt.Orientation.Horizontal)
slider_lower_quadrupole.setTickPosition(QSlider.TickPosition.TicksBelow)
slider_lower_quadrupole.setMinimum(-100)
slider_lower_quadrupole.setMaximum(100)
slider_lower_quadrupole.setValue(0)
slider_lower_quadrupole.setTickPosition(QSlider.TicksBelow)
slider_lower_quadrupole.setTickInterval(100)

lower_quadrupole_label = QLabel('Lower Quadrupole Current (Amperes) = ' + "{:.2f}".format(slider_lower_quadrupole.value()))
lower_quadrupole_label.setMinimumWidth(80)

slider_lens = QSlider(QtCore.Qt.Orientation.Horizontal)
slider_lens.setTickPosition(QSlider.TickPosition.TicksBelow)
slider_lens.setMinimum(0)
slider_lens.setMaximum(5000)
slider_lens.setValue(3000)
slider_lens.setTickPosition(QSlider.TicksBelow)
slider_lens.setTickInterval(100)

lens_label = QLabel('Lens Strength (Tesla) = ' + "{:.2f}".format(slider_lens.value()/1000))
lens_label.setMinimumWidth(80)

vbox = QVBoxLayout()
vbox.addWidget(upper_quadrupole_label)
vbox.addWidget(slider_upper_quadrupole)
vbox.addWidget(lower_quadrupole_label)
vbox.addWidget(slider_lower_quadrupole)
vbox.addWidget(lens_label)
vbox.addWidget(slider_lens)
vbox.addStretch()
groupBox.setLayout(vbox)
d2.addWidget(groupBox, 1, 0)

w3 = gl.GLViewWidget()
d3.addWidget(w3)

scale = 0.01
vertices = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0], [1, 1, 0]]) * scale

square = gl.GLLinePlotItem(pos=vertices, color="w")

w3.addItem(square)

a = gl.GLAxisItem()
w3.addItem(a)
w3.setBackgroundColor("grey")
w3.setCameraPosition(distance=1)
image_size = 128

env = TemGym(3e5, 2**10, 0.014, image_size)
img.setImage(env.detector_image)

ray_lines = dict()
for idx in range(0, env.r.shape[1], 4):
    pts = env.r[:, idx, :]
    ray_lines[idx] = gl.GLLinePlotItem(pos = pts, color=(0, 0, 1, 1))
    w3.addItem(ray_lines[idx])
    
for component in env.component_dict:
    if env.component_dict[component]["Type"] == "Quadrupole":
        w3.addItem(env.component_dict[component]["3D_Mesh_One"])
        w3.addItem(env.component_dict[component]["3D_Mesh_Two"])
        w3.addItem(env.component_dict[component]["3D_Mesh_Three"])
        w3.addItem(env.component_dict[component]["3D_Mesh_Four"])
    elif env.component_dict[component]["Type"] == "Lens":
        w3.addItem(env.component_dict[component]["3D_Mesh"])
        
def update_slider():
    env.upper_quadrupole_bmag = slider_upper_quadrupole.value()
    env.lower_quadrupole_bmag = slider_lower_quadrupole.value()
    env.lens_bmag = slider_lens.value()*1e-3
    
    upper_quadrupole_label.setText(
    'Upper Quadrupole Current (Amperes) = ' + "{:.2f}".format(slider_upper_quadrupole.value()))
    lower_quadrupole_label.setText(
    'Lower Quadrupole Current (Amperes) = ' + "{:.2f}".format(slider_lower_quadrupole.value()))
    lens_label.setText(
    'Lens Strength (Tesla) = ' + "{:.2f}".format(slider_lens.value()*1e-3))

    next_state_frame, reward, done = env.step(1)
    img.setImage(next_state_frame)
    
    for idx in range(0, env.r.shape[1], 4):
        pts = env.r[:, idx, :]
        ray_lines[idx].setData(pos=pts, color=(0, 0, 1, 1))
        
update_slider()
slider_lower_quadrupole.sliderMoved.connect(update_slider)
slider_upper_quadrupole.sliderMoved.connect(update_slider)
slider_lens.sliderMoved.connect(update_slider)
app.exec_()




    


        