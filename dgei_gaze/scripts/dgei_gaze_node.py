#!/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from dgei_interfaces.msg import GazeFrame, GazeDetection

from l2cs.gaze_detectors import Gaze_Detector

import math
import pdb
import cv2
import numpy as np
from time import time, sleep
from collections import deque


