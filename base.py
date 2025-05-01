import os
import cv2
import numpy as np
from zipfile import ZipFile
from urllib.request import urlretrieve
from collections import deque

def __dfs(img, pos, labels, count):
    stack = [pos]  
    labels[pos] = count
    total_pixel = 1  
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        x, y = stack.pop() 

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(img) and 0 <= ny < len(img[0]) and img[x, y] == img[nx, ny] and labels[nx, ny] == 0:
                labels[nx, ny] = count  
                stack.append((nx, ny))  
                total_pixel += 1 

    return total_pixel

def __dfs_object_detection(binary_frame, min_objPixel = 10, max_distance=50):
    rows, cols = binary_frame.shape
    labels = np.zeros((rows,cols), dtype=int)
    
    count = 1
    object = []

    for x in range(rows):
        for y in range(cols):
            if binary_frame[x,y] > 0 and labels[x,y] == 0:
                pixels = __dfs(binary_frame, (x,y), labels, count)    
                if pixels >= min_objPixel:
                    xs, ys = np.where(labels == count)
                    if len(xs) > 0 and len(ys) > 0:
                        min_x, max_x = np.min(xs), np.max(xs)
                        min_y, max_y = np.min(ys), np.max(ys)
                        object.append((min_x, min_y, max_x, max_y,pixels))
                count+=1

    return object

def __bfs(img, pos, labels, count):
    queue = deque([pos])
    labels[pos] = count
    total_pixel = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(img) and 0 <= ny < len(img[0]) and img[x,y] == img[nx,ny] and labels[nx,ny] == 0:
                labels[nx,ny] = count
                queue.append((nx, ny))
                total_pixel +=1

    return total_pixel

def __bfs_object_detection(binary_frame, min_objPixel = 10, max_distance=50):
    rows, cols = binary_frame.shape
    labels = np.zeros((rows,cols), dtype=int)
    
    count = 1
    object = []

    for x in range(rows):
        for y in range(cols):
            if binary_frame[x,y] > 0 and labels[x,y] == 0:
                pixels = __bfs(binary_frame, (x,y), labels, count)    
                if pixels >= min_objPixel:
                    xs, ys = np.where(labels == count)
                    if len(xs) > 0 and len(ys) > 0:
                        min_x, max_x = np.min(xs), np.max(xs)
                        min_y, max_y = np.min(ys), np.max(ys)
                        object.append((min_x, min_y, max_x, max_y,pixels))
                count+=1

    return object

def __tracking_colision(curr_area,pre_obj, tolerance_pixel=20):
    for i in range(len(pre_obj)):
        for j in range(i+1, len(pre_obj)):
            pre_area1 = pre_obj[i][4]
            pre_area2 = pre_obj[j][4]
            total_preArea = pre_area1 + pre_area2
            if total_preArea - tolerance_pixel <= curr_area <= total_preArea + tolerance_pixel:
                return True
    return False

def bfs_obj_collision(video_path, min_objPixel = 10, debug = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
    
    if debug:
        output_folder = 'frames_output'
        os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    previous_objects = []
    while True:
        ret, frame = cap.read()
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if not ret:
            print("Hết video")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 147, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #Tìm đối tượng
        objects = __bfs_object_detection(binary, min_objPixel)
        
        #có sự thay đổi thì tracking
        if len(objects) < len(previous_objects):
            for i in range(len(objects)):
                curr_area = objects[i][4]
                if __tracking_colision(curr_area,previous_objects):
                    print(f"Chạm nhau tại: {current_time_ms}, Frame: {frame_count}")
                    break
        
        previous_objects = objects
            
        if debug:
            debug_frame = frame.copy()
            for obj in objects:
                min_x, min_y, max_x, max_y, area = obj
                cv2.rectangle(debug_frame, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)

            # Lưu khung hình có bounding boxes
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
            cv2.imwrite(frame_filename, debug_frame)
            print(f"Frame {frame_count}: {len(objects)}")

        frame_count+=1

def dfs_obj_collision(video_path, min_objPixel = 10, debug = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
    
    if debug:
        output_folder = 'frames_output'
        os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    previous_objects = []
    while True:
        ret, frame = cap.read()
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if not ret:
            print("Hết video")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 147, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #Tìm đối tượng
        objects = __dfs_object_detection(binary, min_objPixel)
        
        #có sự thay đổi thì tracking
        if len(objects) < len(previous_objects):
            for i in range(len(objects)):
                curr_area = objects[i][4]
                if __tracking_colision(curr_area,previous_objects):
                    print(f"Chạm nhau tại: {current_time_ms}, Frame: {frame_count}")
                    break
        
        previous_objects = objects
            
        if debug:
            debug_frame = frame.copy()
            for obj in objects:
                min_x, min_y, max_x, max_y, area = obj
                cv2.rectangle(debug_frame, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)

            # Lưu khung hình có bounding boxes
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
            cv2.imwrite(frame_filename, debug_frame)
            print(f"Frame {frame_count}: {len(objects)}")

        frame_count+=1