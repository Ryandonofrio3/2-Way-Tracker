import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from collections import deque
import seaborn as sns 
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import json

def process_folder(folder_path, out_video_dir):
    for file in os.listdir(folder_path):
        if any(file.endswith(ext) for ext in [".mov", ".avi", ".mp4"]):
            video_path = os.path.join(folder_path, file)
            process_video(video_path, out_video_dir)
            

def main():
    video_folder_path = r"\\cmontell-nas1.mcdb.ucsb.edu\Ryan\Mosqutio Tracker 2 way choice\Parallel Processing\In Test 1"
    out_video_dir = r"\\cmontell-nas1.mcdb.ucsb.edu\Ryan\Mosqutio Tracker 2 way choice\Parallel Processing\Out Test 1"
    process_folder(video_folder_path, out_video_dir)




def process_video(video_path, out_video_dir):
    video_name = os.path.basename(video_path)
    video = cv2.VideoCapture(video_path)
    out_video_name = video_name.split('.')[0] + '_output.avi'
    out_video_path = os.path.join(out_video_dir, out_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(video.get(cv2.CAP_PROP_FPS))

    def preprocess_frame(frame):
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        return blurred_frame


    roi_points = []


    def select_roi(event, x, y, flags, param, roi_points):

        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            roi_points.append((x, y))
            cv2.rectangle(frame_copy, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
            cv2.imshow('Select Regions of Interest', frame_copy)




    def detect_mosquitoes(frame, rois, min_area=1):
        
        fgmask = fgbg.apply(frame)
        
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        mosquitoes = []

        
        for contour in contours:
            
            if cv2.contourArea(contour) > min_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                mosquito_center = (x + w // 2, y + h // 2)

                
                for roi in rois:
                    if (roi[0][0] <= mosquito_center[0] <= roi[1][0]) and (roi[0][1] <= mosquito_center[1] <= roi[1][1]):
                        mosquitoes.append((x, y, w, h))
                        break

        
        return mosquitoes



    state_cap = int(fps * .6)



    class MosquitoTracker:
        
        def __init__(self, max_frames=state_cap):
            self.positions = deque(maxlen=max_frames)
            self.areas = deque(maxlen=max_frames)
            self.host_seeking_count = 0
            self.time_on_plate1 = 0
            self.time_on_plate2 = 0
            self.distance_on_plate1 = 0
            self.distance_on_plate2 = 0
            self.starting_position_plate1 = None
            self.starting_position_plate2 = None


        
        def update(self, position, on_plate1, on_plate2):
            self.positions.append(position)
            self.areas.append(position)
            
            if on_plate1:
                self.time_on_plate1 += 1
                if self.starting_position_plate1 is None:
                    self.starting_position_plate1 = position
                else:
                    self.distance_on_plate1 = np.linalg.norm(np.array(self.starting_position_plate1) - np.array(position))

            if on_plate2:
                self.time_on_plate2 += 1
                if self.starting_position_plate2 is None:
                    self.starting_position_plate2 = position
                else:
                    self.distance_on_plate2 = np.linalg.norm(np.array(self.starting_position_plate2) - np.array(position))


        
        def is_moving(self, threshold=4.5):
            if len(self.positions) < 2:
                return False

            
            for i in range(len(self.positions) - 1):
                dist = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[i + 1]))
                
                if dist > threshold:
                    self.host_seeking_count += 1
                    return True
            return False



    def track_mosquitoes(mosquitoes, trackers, rois):
        
        for mosquito in mosquitoes:
            x, y, w, h = mosquito
            center = (x + w // 2, y + h // 2)

            
            min_distance = float('inf')
            closest_tracker = None
            on_plate1= False
            on_plate2= False

            for roi in rois:
                if (roi[0][0] <= center[0] <= roi[1][0]) and (roi[0][1] <= center[1] <= roi[1][1]):
                    if rois.index(roi) == 0:
                        on_plate1 = True
                    elif rois.index(roi) == 1:
                        on_plate2 = True
                    break

            for tracker_id, tracker in trackers.items():
                
                distance = np.linalg.norm(np.array(tracker.positions[-1]) - np.array(center))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_tracker = tracker_id

            if closest_tracker is not None and min_distance < 50:
                trackers[closest_tracker].update(center, on_plate1, on_plate2)
            else:
                new_tracker_id = len(trackers)
                trackers[new_tracker_id] = MosquitoTracker()
                trackers[new_tracker_id].update(center, on_plate1, on_plate2)
        return trackers


    fgbg = cv2.createBackgroundSubtractorMOG2()

    frame_count = 0


    host_seeking_count = 0
    trackers = {}


    ret, frame = video.read()
    if ret:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        roi_file = os.path.join(script_dir, 'rois.json') 
        
        frame_copy = frame.copy()

        try:
            with open(roi_file, 'r') as f:
                roi_points = json.load(f)
        except FileNotFoundError:
            roi_points = []
            cv2.namedWindow('Select Regions of Interest')
            cv2.setMouseCallback('Select Regions of Interest', lambda *args: select_roi(*args, roi_points=roi_points))

            while True:
                cv2.imshow('Select Regions of Interest', frame_copy)
                key = cv2.waitKey(1) & 0xFF
                if key == 13:
                    break

            cv2.destroyWindow('Select Regions of Interest')

            with open(roi_file, 'w') as f:
                json.dump(roi_points, f)

        rois = [(roi_points[i], roi_points[i + 1]) for i in range(0, len(roi_points), 2)]

    frame_height, frame_width, _ = frame.shape
    heatmap_p1 = np.zeros((frame_height, frame_width), dtype=np.uint32)
    heatmap_p2 = np.zeros((frame_height, frame_width), dtype=np.uint32)
    trace_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)


    out_video = cv2.VideoWriter(out_video_path, fourcc, 20.0, (frame_width, frame_height))
    walking_events_p1 = 0
    walking_events_p2 = 0
    total_walking_events = 0
    walking_events_p1_list = []
    walking_events_p2_list = []
    total_walking_events_list = []
    hpi_list = []



    while True:
        ret, frame = video.read()
        cv2.putText(frame, 'Video: {}'.format(video_name), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if not ret:
            break

        frame_count += 1
        preprocessed_frame = preprocess_frame(frame)
        
        mosquitoes = detect_mosquitoes(preprocessed_frame, rois)
        trackers = track_mosquitoes(mosquitoes, trackers, rois)
        walking_events_p1_list.append(walking_events_p1)
        walking_events_p2_list.append(walking_events_p2)
        total_walking_events_list.append(total_walking_events)

        
        for tracker in trackers.values():
            if len(tracker.positions) > 1:
                for i in range(len(tracker.positions) - 1):
                    start_point = tuple(map(int, tracker.positions[i]))
                    end_point = tuple(map(int, tracker.positions[i + 1]))
                    color = (0, 255, 0)  
                    cv2.line(trace_image, start_point, end_point, color, 1)
                    

        if total_walking_events > 0:
            proportion_p1 = walking_events_p1 / total_walking_events
            proportion_p2 = walking_events_p2 / total_walking_events
            current_hpi = proportion_p2 - proportion_p1
        else:
            current_hpi = 0




        hpi_list.append(current_hpi)
        for i, (x, y, w, h) in enumerate(mosquitoes):
            tracker = next((t for t in trackers.values() if t.positions[-1] == (x + w // 2, y + h // 2)), None)
            if tracker is not None and tracker.is_moving():
                total_walking_events += 1
                color = (0, 0, 255)  
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if (rois[0][0][0] <= x + w // 2 <= rois[0][1][0]) and (rois[0][0][1] <= y + h // 2 <= rois[0][1][1]):
                    heatmap_p1[y:y+h, x:x+w] += 1
                elif (rois[1][0][0] <= x + w // 2 <= rois[1][1][0]) and (rois[1][0][1] <= y + h // 2 <= rois[1][1][1]):
                    heatmap_p2[y:y+h, x:x+w] += 1

                
                if (rois[0][0][0] <= x + w // 2 <= rois[0][1][0]) and (rois[0][0][1] <= y + h // 2 <= rois[0][1][1]):
                    walking_events_p1 += 1
                
                elif (rois[1][0][0] <= x + w // 2 <= rois[1][1][0]) and (rois[1][0][1] <= y + h // 2 <= rois[1][1][1]):
                    walking_events_p2 += 1



        for roi in rois:
            cv2.rectangle(frame, roi[0], roi[1], (255, 255, 0), 2)

        out_video.write(frame)

        
        cv2.imshow('Mosquito Tracker', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    out_video.release()
    cv2.destroyAllWindows()


    if total_walking_events > 0:
        proportion_p1 = walking_events_p1 / total_walking_events
        proportion_p2 = walking_events_p2 / total_walking_events
        hpi = proportion_p2 - proportion_p1
    else:
        hpi = 0

    
    hsi = total_walking_events / frame_count
    #calculate HSI for each plate

    total_time_on_plate1 = 0
    total_time_on_plate2 = 0


    for tracker in trackers.values():
        total_time_on_plate1 += tracker.time_on_plate1
        total_time_on_plate2 += tracker.time_on_plate2

    total_distance_on_plate1 = 0
    total_distance_on_plate2 = 0
    total_moving_mosquitoes_p1 = 0
    total_moving_mosquitoes_p2 = 0

    for tracker in trackers.values():
        if tracker.host_seeking_count > 0:
            total_distance_on_plate1 += tracker.distance_on_plate1
            total_distance_on_plate2 += tracker.distance_on_plate2
            if tracker.distance_on_plate1 > 0:
                total_moving_mosquitoes_p1 += 1
            if tracker.distance_on_plate2 > 0:
                total_moving_mosquitoes_p2 += 1

    average_distance_on_plate1 = total_distance_on_plate1 / total_moving_mosquitoes_p1 if total_moving_mosquitoes_p1 > 0 else 0
    average_distance_on_plate2 = total_distance_on_plate2 / total_moving_mosquitoes_p2 if total_moving_mosquitoes_p2 > 0 else 0
    
    average_time_on_plate1 = total_time_on_plate1 / total_moving_mosquitoes_p1
    average_time_on_plate2 = total_time_on_plate2 / total_moving_mosquitoes_p2


    corrected_average_time_1 = average_time_on_plate1/fps
    corrected_average_time_2 = average_time_on_plate2/fps

    proportion_walking_events_p1 = walking_events_p1 / total_walking_events if total_walking_events > 0 else 0
    proportion_walking_events_p2 = walking_events_p2 / total_walking_events if total_walking_events > 0 else 0

    walking_event_difference = proportion_walking_events_p2 - proportion_walking_events_p1

    normalized_time_metric = (corrected_average_time_2 - corrected_average_time_1) / (corrected_average_time_1 + corrected_average_time_2)

    normalized_distance_metric = (average_distance_on_plate2 - average_distance_on_plate1) / (average_distance_on_plate1 + average_distance_on_plate2)

    hsi_p1 = walking_events_p1 / frame_count
    hsi_p2 = walking_events_p2 / frame_count





    

    outpath = "41823Test.csv"

    if not os.path.exists(outpath):
        with open(outpath, "w") as f:
            f.write("Video Name,HPI,Total Walk events,Walk Cont, Walk IR, HSI, HSI Cont, HSI IR, Normalized Time, Normalized Distance, Tot Time P1, Tot Time P2, Tot Distance P1, Tot Distance P2\n")


    results = {
        "Video Name": [video_name],
        "HPI": [round(hpi, 6)],
        "Total Walking Events": [total_walking_events],
        "Walks Cont Plate": [walking_events_p1],
        "Walks IR Plate": [walking_events_p2],
        "HSI": [round(hsi, 6)],
        "HSI Control Plate": [round(hsi_p1, 6)],
        "HSI IR Plate": [round(hsi_p2, 6)],
        "Normalized Time Metric": [round(normalized_time_metric, 4)],
        "Normalized Distance Metric": [round(normalized_distance_metric, 4)],
        "Total Time P1": [round(total_time_on_plate1, 4)],
        "Total Time P2": [round(total_time_on_plate2, 4)],
        "Total Distance P1": [round(total_distance_on_plate1, 4)],
        "Total Distance P2": [round(total_distance_on_plate2, 4)],

    }




    df = pd.DataFrame.from_dict(results)


    df.to_csv(outpath, mode="a", index=False, header=False)


    print("Results saved to CSV file:", outpath)


if __name__ == '__main__':
    main()