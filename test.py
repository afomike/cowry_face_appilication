import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, request, jsonify, Response
import sqlite3
import base64
from datetime import datetime
import io
from PIL import Image, ImageOps
import threading
import time
import atexit

def test_image_loading(photo_path):
    try:
        print(f"Testing image: {photo_path}")
        rgb_array = load_rgb_uint8_contiguous(photo_path)
        if rgb_array is None:
            print("❌ Failed to load image")
            return False
            
        print(f"Image shape: {rgb_array.shape}")
        print(f"Image dtype: {rgb_array.dtype}")
        
        # Try face detection
        face_locations = face_recognition.face_locations(rgb_array)
        print(f"Found {len(face_locations)} face(s) in image")
        
        return True
    except Exception as e:
        print(f"❌ Error testing image: {str(e)}")
        return False

# Test your image
test_image_loading("Photo/mike.jpg")