#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:46:50 2022

@author: andy
"""
import sys
import numpy as np
from temgym_beam_shift import TemGym
import PyQt5
from PyQt5 import QtCore, QtWidgets


from PyQt5.QtWidgets import (
    QHBoxLayout,
    QSlider,
    QVBoxLayout,
    QMainWindow,
    QGroupBox,
)

import pyqtgraph.opengl as gl
import pyqtgraph as pg

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
groupBox = QGroupBox("Upper Deflector Deflection")

slider_saddle_one = QSlider(QtCore.Qt.Orientation.Horizontal)
slider_saddle_one.setTickPosition(QSlider.TickPosition.TicksBelow)
slider_saddle_one.setMinimum(-200)
slider_saddle_one.setMaximum(200)
slider_saddle_one.setValue(0)
slider_saddle_one.setTickPosition(QSlider.TicksBelow)
slider_saddle_one.setTickInterval(100)

vbox = QVBoxLayout()
vbox.addWidget(slider_saddle_one)
groupBox.setLayout(vbox)
d2.addWidget(groupBox, 1, 0)

groupBox = QGroupBox("Deflector Ratio Nudge (Coarse and Fine)")

slider_saddle_two_button_coarse_left = QtWidgets.QToolButton()
slider_saddle_two_button_coarse_left.setIconSize(QtCore.QSize(100, 100))
slider_saddle_two_button_coarse_left.setArrowType(QtCore.Qt.LeftArrow)

slider_saddle_two_button_left = QtWidgets.QToolButton()
slider_saddle_two_button_left.setIconSize(QtCore.QSize(100, 100))
slider_saddle_two_button_left.setArrowType(QtCore.Qt.LeftArrow)


slider_saddle_two_button_right = QtWidgets.QToolButton()
slider_saddle_two_button_right.setIconSize(QtCore.QSize(100, 100))
slider_saddle_two_button_right.setArrowType(QtCore.Qt.RightArrow)

slider_saddle_two_button_coarse_right = QtWidgets.QToolButton()
slider_saddle_two_button_coarse_right.setIconSize(QtCore.QSize(100, 100))
slider_saddle_two_button_coarse_right.setArrowType(QtCore.Qt.RightArrow)

hbox = QHBoxLayout()
hbox.addWidget(slider_saddle_two_button_coarse_left)
hbox.addWidget(slider_saddle_two_button_left)
hbox.addWidget(slider_saddle_two_button_right)
hbox.addWidget(slider_saddle_two_button_coarse_right)
vbox = QVBoxLayout()
vbox.addLayout(hbox)
vbox.addStretch()
groupBox.setLayout(vbox)
d2.addWidget(groupBox, 2, 0)

w3 = gl.GLViewWidget()
d3.addWidget(w3)

gx = gl.GLGridItem()
gx.rotate(90, 0, 1, 0)
gx.translate(-10, 0, 0)
w3.addItem(gx)
gy = gl.GLGridItem()
gy.rotate(90, 1, 0, 0)
gy.translate(0, -10, 0)
w3.addItem(gy)
gz = gl.GLGridItem()
gz.translate(0, 0, -10)

scale = 0.01
vertices = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0], [1, 1, 0]]) * scale

square = gl.GLLinePlotItem(pos=vertices, color="w")

w3.addItem(square)

a = gl.GLAxisItem()
w3.addItem(a)
w3.setBackgroundColor("gray")
w3.setCameraPosition(distance=5)
ray_lines = gl.GLLinePlotItem(mode = 'line_strip')
w3.addItem(ray_lines)
image_size = 128

env = TemGym(3e5, 2**9, 0.014, image_size)
img.setImage(env.detector_image)

ray_lines = dict()
for idx in range(0, env.r.shape[1], 8):
    pts = env.r[:, idx, :]
    ray_lines[idx] = gl.GLLinePlotItem(pos = pts, color=(0, 0, 1, 1))
    w3.addItem(ray_lines[idx])
    
for component in env.component_dict:
    if env.component_dict[component]["Type"] == "Saddle":
        w3.addItem(env.component_dict[component]["3D_Mesh_One"])
        w3.addItem(env.component_dict[component]["3D_Mesh_Two"])
    elif env.component_dict[component]["Type"] == "Lens":
        w3.addItem(env.component_dict[component]["3D_Mesh"])
    
def update_slider():
    env.saddle_one_bmag = slider_saddle_one.value()*1e-1
    next_state_frame, reward, done = env.step(2)
    # print(env.saddle_two_ratio)
    img.setImage(next_state_frame)
    
    for idx in range(0, env.r.shape[1], 8):
        pts = env.r[:, idx, :]
        ray_lines[idx].setData(pos=pts, color=(0, 0, 1, 1))
        
    # print(env.average_positions[-1])
    
def update_button_left():
    # env.saddle_two_ratio -= 0.05
    
    next_state_frame, reward, done = env.step(1)
    print('Def Ratio = ', env.saddle_two_ratio)
    # print('Reward = ', reward)
    print('Position = ', env.average_positions[-1])
    # print('done = ', done)
    img.setImage(next_state_frame)
    # print(env.average_positions[-1])
    
    for idx in range(0, env.r.shape[1], 8):
        pts = env.r[:, idx, :]
        ray_lines[idx].setData(pos=pts, color=(0, 0, 1, 1))
        
def update_button_right():
    # env.saddle_two_ratio += 0.05
    
    next_state_frame, reward, done = env.step(3)
    print('Def Ratio = ', env.saddle_two_ratio)
    # print('Reward = ', reward)
    print('Position = ', env.average_positions[-1])
    # print('done = ', done)
    img.setImage(next_state_frame)
    # print(env.average_positions[-1])
    
    for idx in range(0, env.r.shape[1], 8):
        pts = env.r[:, idx, :]
        ray_lines[idx].setData(pos=pts, color=(0, 0, 1, 1))
    
def update_button_coarse_left():
    # env.saddle_two_ratio -= 0.2
    # env.lens_bmag -=0.1
    next_state_frame, reward, done = env.step(0)
    print('Def Ratio = ', env.saddle_two_ratio)
    # print('Reward = ', reward)
    print('Position = ', env.average_positions[-1])
    # print('done = ', done)
    img.setImage(next_state_frame)
    # print(env.average_positions[-1])
    
    for idx in range(0, env.r.shape[1], 8):
        pts = env.r[:, idx, :]
        ray_lines[idx].setData(pos=pts, color=(0, 0, 1, 1))
    
def update_button_coarse_right():
    # env.saddle_two_ratio += 0.2
    # env.lens_bmag +=0.1
    next_state_frame, reward, done = env.step(4)
    print('Def Ratio = ', env.saddle_two_ratio)
    # print('Reward = ', reward)
    print('Position = ', env.average_positions[-1])
    # print('done = ', done)
    img.setImage(next_state_frame)
    # print(env.average_positions[-1])
    
    for idx in range(0, env.r.shape[1], 8):
        pts = env.r[:, idx, :]
        ray_lines[idx].setData(pos=pts, color=(0, 0, 1, 1))
    
update_slider()
slider_saddle_one.sliderMoved.connect(update_slider)
slider_saddle_two_button_left.clicked.connect(update_button_left)
slider_saddle_two_button_right.clicked.connect(update_button_right)
slider_saddle_two_button_coarse_left.clicked.connect(update_button_coarse_left)
slider_saddle_two_button_coarse_right.clicked.connect(update_button_coarse_right)

app.exec_()




    


        