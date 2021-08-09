import cv2
from matplotlib import pyplot as plt
import json
from sensecam_control import onvif_control
from sensecam_control import onvif_config
import pandas as pd
import time
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np

RTSP1 = 'rtsp://admin:Hikvisionarv1234@192.168.1.64'
ip = '192.168.1.64'
login = 'arvonvif'
password = 'Arvonvif1234'
cam = onvif_control.CameraControl(ip, login, password)
cam.camera_start()
pan_shift = 0.00556

pts_dict = {
	"ptz_manual":{
		"home": (0.934, 0.505818, 0.0),
		"point1":(0.932556, 0.230364, 0.051429),
		"point2":(0.999333, 0.230364, 0.051429),
		"point3":(-0.943167, 0.230364, 0.051429),
		"point4":(0.923167, 0.340909, 0.077143),
		"point5":(0.990167, 0.340909, 0.077143),
		"point6":(-0.937333, 0.340909, 0.077143),
		"point7":(0.868, 0.404909, 0.116857),
		"point8":(0.9245, 0.404909, 0.114286),
		"point9":(0.983611, 0.404909, 0.114286),
		"point10":(-0.93, 0.404909, 0.114286),
		"point11":(0.859556, 0.448182, 0.145714),
		"point12":(0.918722, 0.448182, 0.142857),
		"point13":(0.982, 0.448182, 0.14),
		"point14":(-0.966722, 0.448182, 0.14),
		"point15":(0.869556, 0.480364, 0.177143),
		"point16":(0.918056, 0.480364, 0.177143),
		"point17":(0.974611, 0.480364, 0.177143),
		"point18":(-0.983722, 0.480364, 0.177143),
		"point19":(0.8835, 0.505818, 0.205429),
		"point20":(0.919056, 0.505818, 0.205429),
		"point21":(0.962056, 0.505818, 0.205429),
		"point22":(0.896889, 0.522727, 0.245714),
		"point23":(0.940667, 0.522727, 0.242857),
		"point24":(0.894167, 0.532909, 0.271429),
		"point25":(0.926667, 0.532909, 0.271429),
		"point26":(0.903, 0.546364, 0.311429),
		"point27":(0.925833, 0.546364, 0.311429),
		"point28":(0.908, 0.554909, 0.337143),
		"point29":(0.920944, 0.554909, 0.337143)
		},

	"xy_pts":{'alignment': 
		[(1265,1373),(1743,1384),(2175,1429),(1196,1116),(1680,1120),(2232,1161),(792,982),(1204,970),(1636,973),(2306,1013),
		 (724,889),(1164,873),(1629,876),(2016,890),(799,813),(1159,803),(1572,804),(1883,813),(901,751),(1161,747),
		 (1486,746),(1005,711),(1328,707),(985,689),(1221,687),(1050,657),(1217,655),(1086,638),(1182,638)]
		}
}

def cap_frame(inRTSP):
    capture = cv2.VideoCapture(inRTSP)
    ret, frame = capture.read()
    return ret, frame

def draw_midline(in_frame):
    out_frame = in_frame.copy()
    height, width, _ = out_frame.shape
    half_height = int(height / 2)
    half_width = int(width / 2)
    out_frame = cv2.line(out_frame, (half_width, 0),(half_width, height), color=(255,0,0), thickness=3)
    out_frame = cv2.line(out_frame, (0, half_height), (width, half_height), color=(255, 0, 0), thickness=3)
    return out_frame

def see_center(inRTSP):
    _, frame = cap_frame(inRTSP)
    frame = frame[:,:,[2,1,0]]
    out_frame = draw_midline(frame)
    return out_frame

def draw_alignment_pts(in_frame, list_pts):
    out_frame = in_frame.copy()
    for point in list_pts:
        x,y = point
        out_frame = cv2.circle(out_frame, (x,y), radius=3, color=(0,0,255),thickness=-1)
    return out_frame

def run_pts_check():
    folder = datetime.now().strftime("alignment_%d%m%Y_%H%M%S")
    os.mkdir(folder)

    pan, tilt, zoom = pts_dict["ptz_manual"]['home']
    cam.absolute_move(pan, tilt, zoom)
    time.sleep(20)
    _, frame = cap_frame(RTSP1)
    z0_align = draw_alignment_pts(frame, pts_dict['xy_pts']['alignment'])
    cv2.imwrite(os.path.join(folder, "home_pts_align" + ".jpg"), z0_align)

    list_points = pts_dict["ptz_manual"]
    ptz_state_measured = {}
    for key in tqdm(list_points.keys()):
        pan, tilt, zoom = pts_dict["ptz_manual"][key]
        if pan < 0:
            pan = pan + pan_shift    
        cam.absolute_move(pan, tilt, zoom)
        time.sleep(20)
        _, frame = cap_frame(RTSP1)
        frame2 = see_center(RTSP1)
        cv2.imwrite(os.path.join(folder, key + ".jpg"), frame2[:,:,[2,1,0]])
        m_pan, m_tilt, m_zoom = cam.get_ptz()
        ptz_state_measured[key] = [m_pan, m_tilt, m_zoom]

    with open(os.path.join(folder, "measured_ptz"), 'w') as fp:
        json.dump(ptz_state_measured, fp)

# run_pts_check()

PTZ_MAP = np.array([[0.932556, 0.230364, 0.051429],
                     [0.999333, 0.230364, 0.051429],
                     [-0.943167, 0.230364, 0.051429],
                     [0.923167, 0.340909, 0.077143],
                     [0.990167, 0.340909, 0.077143],
                     [-0.937333, 0.340909, 0.077143], 
                     [0.868, 0.404909, 0.116857], 
                     [0.9245, 0.404909, 0.116857], 
                     [0.983611, 0.404909, 0.116857], 
                     [-0.93, 0.404909, 0.116857], 
                     [0.859556, 0.448182, 0.145714], 
                     [0.918722, 0.448182, 0.145714], 
                     [0.982, 0.448182, 0.145714], 
                     [-0.966722, 0.448182, 0.145714], 
                     [0.869556, 0.480364, 0.177143], 
                     [0.918056, 0.480364, 0.177143], 
                     [0.974611, 0.480364, 0.177143], 
                     [-0.983722, 0.480364, 0.177143], 
                     [0.8835, 0.505818, 0.205429], 
                     [0.919056, 0.505818, 0.20542], 
                     [0.962056, 0.505818, 0.205429], 
                     [0.896889, 0.522727, 0.245714], 
                     [0.940667, 0.522727, 0.242857], 
                     [0.894167, 0.532909, 0.271429], 
                     [0.926667, 0.532909, 0.271429], 
                     [0.903, 0.546364, 0.311429],
                     [0.925833, 0.546364, 0.311429],
                     [0.908, 0.554909, 0.337143],
                     [0.920944, 0.554909, 0.337143]])

XY_MAP = np.array([[1265,1373],[1743,1384],[2175,1429],[1196,1116],[1680,1120],[2232,1161],[792,982],[1204,970],[1636,973],[2306,1013],
		            [724,889],[1164,873],[1629,876],[2016,890],[799,813],[1159,803],[1572,804],[1883,813],[901,751],[1161,747],
		            [1486,746],[1005,711],[1328,707],[985,689],[1221,687],[1050,657],[1217,655],[1086,638],[1182,638]])

Y_AVG = np.array([1395.3, 1132.3, 984.5, 882, 808.25, 748, 709, 688, 656, 638])
Z_LEVEL = np.array([0.051429, 0.077143, 0.116857, 0.145714, 0.177143, 0.205429, 0.245714, 0.271429, 0.311429, 0.337143])

IDX2Z = [[0,1,2],
         [3,4,5],
         [6,7,8,9],
         [10,11,12,13],
         [14,15,16,17],
         [18,19,20],
         [21,22],
         [23,24],
         [25,26],
         [27,28]
         ]

def find_zlevel(in_pt_yx):
    y, x = in_pt_yx
    y_fiducial = XY_MAP[:, 1]
    print(y_fiducial)
    diff = np.abs(np.subtract(y_fiducial, y))
    idx_min = np.argmin(diff)
    z = PTZ_MAP[idx_min,2]
    # print("idx_min:", idx_min)
    # print("zoom:",z)
    return idx_min, z
    
def zoom_interpolate(in_pt_yx):
    y, x = in_pt_yx
    diff = np.abs(np.subtract(Y_AVG, y))
    idx_min = np.argmin(diff)
    idx_sorted = sorted(range(len(diff)), key=lambda k: diff[k])
    min1 = Y_AVG[idx_sorted[0]]
    min2 = Y_AVG[idx_sorted[1]]
    z1 = Z_LEVEL[idx_sorted[0]]
    z2 = Z_LEVEL[idx_sorted[1]]
    # print("min1 and min2 is: ",min1," & " ,min2)
    # print("z1 and z2 is: ",z1," & " ,z2)
    gap = abs(min1 - min2)
    least_y = min(min1, min2)
    least_z = min(z1, z2)
    ratio = (y - least_y) / gap
    new_z = least_z + (ratio * abs(z1 - z2))
    return idx_min, new_z



def pan_interpolate(idx_zoom_level, in_pt_yx):
    xOne = 1757
    y, x = in_pt_yx
    list_idxer = IDX2Z[idx_zoom_level]
    x_ = np.array([XY_MAP[x_, 0] for x_ in list_idxer])
    pan_ = np.array([PTZ_MAP[x_, 0] for x_ in list_idxer])
    diff = np.abs(np.subtract(x,x_))
    nearest_pairs_x = [x_[i] for i in np.argsort(diff)[:2]]
    nearest_pairs_pan = [pan_[i] for i in np.argsort(diff)[:2]]
    x_1, x_2 = np.sort(nearest_pairs_x)
    pan1, pan2 = np.sort(nearest_pairs_pan)
    if (pan1 < 0 and pan2 < 0) or (pan1 > 0 and pan2 > 0):
        gap = abs(x_1-x_2)
        ratio = (x - np.min(nearest_pairs_x)) / gap
        new_pan = pan1 + (ratio * (pan2 - pan1))
    else:
        if x > xOne:
            x_1 = xOne
            pan2 = pan1
            pan1 = -1.0
        if x <= xOne:
            x_2 = xOne
            pan1 = pan2
            pan2 = 1.0
        gap = abs(x_1-x_2)
        ratio = (x - x_1) / gap
        new_pan = pan1 + (ratio * (pan2 - pan1))    

    # print("X1 and X2 is: ",x_1," & " ,x_2)
    # print("pan1 and pan2 is: ",pan1," & " ,pan2)
    if new_pan < 0:
        new_pan += pan_shift
    return new_pan


def tilt_interpolate(in_pt_yx):
    y, x = in_pt_yx
    diff = np.abs(np.subtract(Y_AVG, y))
    tilt_list = np.unique(PTZ_MAP[:, 1])
    nearest_pairs_tilt = [tilt_list[i] for i in np.argsort(diff)[:3:2]]
    nearest_pairs_y = [Y_AVG[i] for i in np.argsort(diff)[:3:2]]
    y_1, y_2 = nearest_pairs_y
    # print("nearest tilt", nearest_pairs_tilt)
    # print("y1 and y2 is: ",y_1," & " ,y_2)
    gap = abs(y_1 - y_2)
    ratio = 1 - (y - np.min(nearest_pairs_y)) / gap
    tilt1, tilt2 = np.sort(nearest_pairs_tilt)
    new_tilt = tilt1 + (ratio * abs(tilt1 - tilt2))
    return new_tilt



### Mutiple point interpolate ###
lis = [(1310,2019), (912,849), (873,1083), (933,1386), (1012,2180),
        (783,916), (806,1435), (751,1376), (719,1090), (642,1131)]
ptz = []
for i in range(len(lis)):
    PTyx = (lis[i])
    idx_min, z = zoom_interpolate(PTyx)
    p = pan_interpolate(idx_min, PTyx)
    t = tilt_interpolate(PTyx)
    r = [p,t,z]
    ptz.append(r)
    cam.absolute_move(p, t, z)
    time.sleep(20)
    cam.get_ptz()
    result = see_center(RTSP1)
    R = result[:,:,[2,1,0]]
    cv2.imwrite('RandomPoint%s.jpg' %(i+1), R)
print(ptz)

### Single point interpolate ###
PTyx = (1310,2019)
idx_min, z = zoom_interpolate(PTyx)
p = pan_interpolate(idx_min, PTyx)
t = tilt_interpolate(PTyx)

print(p, t, z)
cam.absolute_move(p, t, z)
time.sleep(20)
cam.get_ptz()
result = see_center(RTSP1)
plt.imshow(result)

### Precision test ###

o = [[-0.9643774019138756, 0.2697285477075589, 0.06822881333333333],
 [0.8765506757018393, 0.43871670588235295, 0.122868875],
 [0.9083647086156825, 0.4547931527777778, 0.17036419607843137],
 [0.9495598025169409, 0.42988243137254906, 0.12847995833333334],
 [-0.9460655737704918, 0.39832743971631207, 0.14958831944444445],
 [0.8854870521920668, 0.4925840961538462, 0.19227272093023257],
 [0.9561555657620042, 0.4833087581699346, 0.2023592015503876],
 [0.9472847863247863, 0.5048314108527132, 0.24826368354430378],
 [0.9084095263157895, 0.5178784285714286, 0.2156277341772152],
 [0.9140675, 0.553149, 0.31714322222222224]]
Precision = []

for i in range(len(o)):
    p,t,z = o[i]
    a,b,c = [0.934, 0.505818, 0.0]
    for j in range(3):
        cam.absolute_move(a,b,c)
        time.sleep(10)
        cam.absolute_move(p, t, z)
        time.sleep(20)
        B = cam.get_ptz()
        Precision.append(B)
        result = see_center(RTSP1)
        R = result[:,:,[2,1,0]]
        cv2.imwrite('RandomPoint%s_Precision%s.jpg' %((i+1),j), R)
        if j == 0:
            a,b,c = [0.24, 0.505818, 0.2]
        if j == 1:
            a,b,c = [-0.54, 0.24, 0.0]
        if j == 2:
            a,b,c = [0.934, 0.80, 0.6]
print(Precision)

