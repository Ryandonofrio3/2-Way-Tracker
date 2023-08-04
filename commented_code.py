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
    video_folder_path = r""
    out_video_dir = r""
    process_folder(video_folder_path, out_video_dir)




def process_video(video_path, out_video_dir):
    video_name = os.path.basename(video_path)
    video = cv2.VideoCapture(video_path)
    out_video_name = video_name.split('.')[0] + '_output.avi'
    out_video_path = os.path.join(out_video_dir, out_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(video.get(cv2.CAP_PROP_FPS))

    def preprocess_frame(frame):
        # Convert the frame from BGR to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to the grayscale frame to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        
        # Return the blurred frame
        return blurred_frame


    roi_points = [] # Initialize an empty list to store the points of the Region of Interest


    def select_roi(event, x, y, flags, param, roi_points):
        # Check if the left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append the point to the list
            roi_points.append((x, y))
        # Check if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # Append the point to the list
            roi_points.append((x, y))
            # Draw a rectangle on the frame_copy using the last two points in the list
            cv2.rectangle(frame_copy, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
            # Show the frame
            cv2.imshow('Select Regions of Interest', frame_copy)


    def detect_mosquitoes(frame, rois, min_area=1):
        # Apply the foreground-background subtractor to the frame
        fgmask = fgbg.apply(frame)
        
        # Threshold the foreground mask to create a binary image
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize an empty list to store the detected mosquitoes
        mosquitoes = []

        # Iterate over each contour
        for contour in contours:
            # Check if the area of the contour is greater than the minimum area
            if cv2.contourArea(contour) > min_area:
                # Get the bounding rectangle of the contour
                (x, y, w, h) = cv2.boundingRect(contour)
                # Calculate the center of the bounding rectangle
                mosquito_center = (x + w // 2, y + h // 2)

                # Iterate over each Region of Interest
                for roi in rois:
                    # Check if the center of the bounding rectangle is inside the Region of Interest
                    if (roi[0][0] <= mosquito_center[0] <= roi[1][0]) and (roi[0][1] <= mosquito_center[1] <= roi[1][1]):
                        # If it is, append the bounding rectangle to the list of mosquitoes
                        mosquitoes.append((x, y, w, h))
                        # Break out of the inner loop
                        break

            return mosquitoes



    state_cap = int(fps * .8)



    class MosquitoTracker:
        def __init__(self, max_frames=state_cap):
            # Initialize two deques to store the positions and areas of the mosquito
            self.positions = deque(maxlen=max_frames)
            self.areas = deque(maxlen=max_frames)
            # Initialize the variables to store the number of times the mosquito has sought out a host
            self.host_seeking_count = 0
            # Initialize the variables to store the time spent by the mosquito on plate 1 and plate 2
            self.time_on_plate1 = 0
            self.time_on_plate2 = 0
            # Initialize the variables to store the distance traveled by the mosquito on plate 1 and plate 2
            self.distance_on_plate1 = 0
            self.distance_on_plate2 = 0
            # Initialize the variables to store the starting position of the mosquito on plate 1 and plate 2
            self.starting_position_plate1 = None
            self.starting_position_plate2 = None

        def update(self, position, on_plate1, on_plate2):
            # Append the position and area to the respective deques
            self.positions.append(position)
            self.areas.append(position)
            
            # Check if the mosquito is on plate 1
            if on_plate1:
                # Increment the time spent on plate 1
                self.time_on_plate1 += 1
                # Check if the starting position on plate 1 has not been set
                if self.starting_position_plate1 is None:
                    # If it hasn't, set it to the current position
                    self.starting_position_plate1 = position
                else:
                    # If it has, calculate the distance traveled on plate 1
                    self.distance_on_plate1 = np.linalg.norm(np.array(self.starting_position_plate1) - np.array(position))
            
            # Check if the mosquito is on plate 2
            if on_plate2:
                # Increment the time spent on plate 2
                self.time_on_plate2 += 1
                # Check if the starting position on plate 2 has not been set
                if self.starting_position_plate2 is None:
                    # If it hasn't, set it to the current position
                    self.starting_position_plate2 = position
                else:
                    # If it has, calculate the distance traveled on plate 2
                    self.distance_on_plate2 = np.linalg.norm(np.array(self.starting_position_plate2) - np.array(position))

        
        def is_moving(self, threshold=4.5):
            # Check if there are at least two positions in the deque
            if len(self.positions) < 2:
                return False

            # Iterate over each pair of positions
            for i in range(len(self.positions) - 1):
                # Calculate the distance between the two positions
                dist = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[i + 1]))
                
                # Check if the distance is greater than the threshold
                if dist > threshold:
                    # If it is, increment the host-seeking count
                    self.host_seeking_count += 1
                    # Return True, indicating that the mosquito is moving
                    return True
            # Return False, indicating that the mosquito is not moving
            return False




    def track_mosquitoes(mosquitoes, trackers, rois):
        # Iterate over each detected mosquito
        for mosquito in mosquitoes:
            # Get the center of the bounding rectangle
            x, y, w, h = mosquito
            center = (x + w // 2, y + h // 2)

            # Initialize variables to store the minimum distance and closest tracker
            min_distance = float('inf')
            closest_tracker = None
            # Initialize variables to store whether the mosquito is on plate 1 or plate 2
            on_plate1= False
            on_plate2= False

            # Iterate over each Region of Interest
            for roi in rois:
                # Check if the center of the bounding rectangle is inside the Region of Interest
                if (roi[0][0] <= center[0] <= roi[1][0]) and (roi[0][1] <= center[1] <= roi[1][1]):
                    # If it is, set the appropriate flag
                    if rois.index(roi) == 0:
                        on_plate1 = True
                    elif rois.index(roi) == 1:
                        on_plate2 = True
                    # Break out of the loop
                    break

            # Iterate over each tracker
            for tracker_id, tracker in trackers.items():
                # Calculate the distance between the center of the bounding rectangle and the last position of the tracker
                distance = np.linalg.norm(np.array(tracker.positions[-1]) - np.array(center))
                
                # Check if the distance is less than the minimum distance
                if distance < min_distance:
                    # If it is, update the minimum distance and closest tracker
                    min_distance = distance
                    closest_tracker = tracker_id

            # Check if there is a closest tracker and the minimum distance is less than 50
            if closest_tracker is not None and min_distance < 50:
                # If there is, update the closest tracker
                trackers[closest_tracker].update(center, on_plate1, on_plate2)
            else:
                # If there isn't, create a new tracker
                new_tracker_id = len(trackers)
                trackers[new_tracker_id] = MosquitoTracker()
                trackers[new_tracker_id].update(center, on_plate1, on_plate2)
        # Return the updated trackers
        return trackers


        # Create a Background Subtractor MOG2 object
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Initialize a counter to keep track of the number of frames
    frame_count = 0

    # Initialize a variable to store the number of times a mosquito has sought out a host
    host_seeking_count = 0
    # Initialize a dictionary to store the trackers
    trackers = {}

    # Read a frame from the video
    ret, frame = video.read()

    # Check if a frame was successfully read
    if ret:
        # Get the directory of the script
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Create the file path for the Region of Interest file
        roi_file = os.path.join(script_dir, 'rois.json') 
        
        # Make a copy of the frame
        frame_copy = frame.copy()

        # Try to open the Region of Interest file
        try:
            with open(roi_file, 'r') as f:
                roi_points = json.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, initialize an empty list of Region of Interest points
            roi_points = []
            # Create a window to select the Regions of Interest
            cv2.namedWindow('Select Regions of Interest')
            # Set the mouse callback for the window
            cv2.setMouseCallback('Select Regions of Interest', lambda *args: select_roi(*args, roi_points=roi_points))

            # Show the window until the Enter key is pressed
            while True:
                cv2.imshow('Select Regions of Interest', frame_copy)
                key = cv2.waitKey(1) & 0xFF
                if key == 13:
                    break

            # Destroy the window
            cv2.destroyWindow('Select Regions of Interest')

            # Write the Region of Interest points to the file
            with open(roi_file, 'w') as f:
                json.dump(roi_points, f)

        # Create a list of Regions of Interest as pairs of points
        rois = [(roi_points[i], roi_points[i + 1]) for i in range(0, len(roi_points), 2)]


    # Get the height and width of the frame
    frame_height, frame_width, _ = frame.shape

    # Initialize numpy arrays to store the heatmaps for plate 1 and plate 2
    heatmap_p1 = np.zeros((frame_height, frame_width), dtype=np.uint32)
    heatmap_p2 = np.zeros((frame_height, frame_width), dtype=np.uint32)
    # Initialize a numpy array to store the trace image
    trace_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Initialize a VideoWriter object to write the output video
    out_video = cv2.VideoWriter(out_video_path, fourcc, 20.0, (frame_width, frame_height))

    # Initialize variables to store the number of walking events on plate 1, plate 2, and total
    walking_events_p1 = 0
    walking_events_p2 = 0
    total_walking_events = 0

    # Initialize lists to store the walking events on plate 1, plate 2, total, and Host-seeking Probability Index (HPI)
    walking_events_p1_list = []
    walking_events_p2_list = []
    total_walking_events_list = []
    hpi_list = []



    while True:
        # Read a frame from the video
        ret, frame = video.read()
        # Add text to the frame to indicate the name of the video
        cv2.putText(frame, 'Video: {}'.format(video_name), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # If there are no more frames in the video, break the loop
        if not ret:
            break
        
        # Increment the frame count
        frame_count += 1
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        
        # Detect mosquitoes in the frame
        mosquitoes = detect_mosquitoes(preprocessed_frame, rois)
        # Track the mosquitoes
        trackers = track_mosquitoes(mosquitoes, trackers, rois)
        # Add the number of walking events on plate 1 and plate 2 to their respective lists
        walking_events_p1_list.append(walking_events_p1)
        walking_events_p2_list.append(walking_events_p2)
        # Add the total number of walking events to the list of total walking events
        total_walking_events_list.append(total_walking_events)

        # Draw lines to trace the movement of mosquitoes
        for tracker in trackers.values():
            # Check if the tracker has more than one position
            if len(tracker.positions) > 1:
                # Draw lines between consecutive positions of the tracker
                for i in range(len(tracker.positions) - 1):
                    start_point = tuple(map(int, tracker.positions[i]))
                    end_point = tuple(map(int, tracker.positions[i + 1]))
                    color = (0, 255, 0)  
                    cv2.line(trace_image, start_point, end_point, color, 1)

        # Calculate the Host-seeking Probability Index (HPI) for the current frame
        if total_walking_events > 0:
            # Calculate the proportion of walking events on plate 1 and plate 2
            proportion_p1 = walking_events_p1 / total_walking_events
            proportion_p2 = walking_events_p2 / total_walking_events
            # Calculate the HPI as the difference between the proportions of walking events on plate 2 and plate 1
            current_hpi = proportion_p2 - proportion_p1
        else:
            # If there are no walking events, set the HPI to 0
            current_hpi = 0




        hpi_list.append(current_hpi)
    for i, (x, y, w, h) in enumerate(mosquitoes):
        # Find the tracker that corresponds to the current mosquito
        tracker = next((t for t in trackers.values() if t.positions[-1] == (x + w // 2, y + h // 2)), None)
        
        # If the tracker exists and the mosquito is moving
        if tracker is not None and tracker.is_moving():
            # Increment the total number of walking events
            total_walking_events += 1
            
            # Draw a rectangle around the mosquito
            color = (0, 0, 255)  
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # If the mosquito is in the first region of interest
            if (rois[0][0][0] <= x + w // 2 <= rois[0][1][0]) and (rois[0][0][1] <= y + h // 2 <= rois[0][1][1]):
                # Increment the heatmap for the first region of interest
                heatmap_p1[y:y+h, x:x+w] += 1
            
            # If the mosquito is in the second region of interest
            elif (rois[1][0][0] <= x + w // 2 <= rois[1][1][0]) and (rois[1][0][1] <= y + h // 2 <= rois[1][1][1]):
                # Increment the heatmap for the second region of interest
                heatmap_p2[y:y+h, x:x+w] += 1
            
            # If the mosquito is in the first region of interest
            if (rois[0][0][0] <= x + w // 2 <= rois[0][1][0]) and (rois[0][0][1] <= y + h // 2 <= rois[0][1][1]):
                # Increment the number of walking events in the first region of interest
                walking_events_p1 += 1
            
            # If the mosquito is in the second region of interest
            elif (rois[1][0][0] <= x + w // 2 <= rois[1][1][0]) and (rois[1][0][1] <= y + h // 2 <= rois[1][1][1]):
                # Increment the number of walking events in the second region of interest
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

    # Calculate the proportion of walking events in each region of interest
    if total_walking_events > 0:
        proportion_p1 = walking_events_p1 / total_walking_events
        proportion_p2 = walking_events_p2 / total_walking_events
            # Calculate the heat preference index (HPI) as the difference between the proportions
        hpi = proportion_p2 - proportion_p1
    else:
            # If there are no walking events, set the HPI to zero
        hpi = 0

    # Calculate the heat sensation index (HSI) as the total number of walking events divided by the number of frames
    hsi = total_walking_events / frame_count

    # Calculate the total time spent by mosquitoes on each plate
    total_time_on_plate1 = 0
    total_time_on_plate2 = 0
    for tracker in trackers.values():
        total_time_on_plate1 += tracker.time_on_plate1
        total_time_on_plate2 += tracker.time_on_plate2

    # Initialize variables to keep track of the total distance traveled by mosquitoes on each plate
    total_distance_on_plate1 = 0
    total_distance_on_plate2 = 0

    # Initialize variables to keep track of the number of moving mosquitoes on each plate
    total_moving_mosquitoes_p1 = 0
    total_moving_mosquitoes_p2 = 0


    # Calculate the average distance of all mosquitoes on each plate
    for tracker in trackers.values():
        # If the mosquito is host seeking
        if tracker.host_seeking_count > 0:
            # Sum up the distance of the mosquito on each plate
            total_distance_on_plate1 += tracker.distance_on_plate1
            total_distance_on_plate2 += tracker.distance_on_plate2
            
            # If the mosquito has moved on plate 1, increment the total number of moving mosquitoes on plate 1
            if tracker.distance_on_plate1 > 0:
                total_moving_mosquitoes_p1 += 1
            
            # If the mosquito has moved on plate 2, increment the total number of moving mosquitoes on plate 2
            if tracker.distance_on_plate2 > 0:
                total_moving_mosquitoes_p2 += 1

    # Calculate the average distance of mosquitoes on plate 1
    average_distance_on_plate1 = total_distance_on_plate1 / total_moving_mosquitoes_p1 if total_moving_mosquitoes_p1 > 0 else 0

    # Calculate the average distance of mosquitoes on plate 2
    average_distance_on_plate2 = total_distance_on_plate2 / total_moving_mosquitoes_p2 if total_moving_mosquitoes_p2 > 0 else 0

    # Calculate the average time that mosquitoes spent on each plate
    average_time_on_plate1 = total_time_on_plate1 / total_moving_mosquitoes_p1
    average_time_on_plate2 = total_time_on_plate2 / total_moving_mosquitoes_p2

    # Correct the average time for each plate by dividing by fps
    corrected_average_time_1 = average_time_on_plate1 / fps
    corrected_average_time_2 = average_time_on_plate2 / fps

    # Calculate the proportion of walking events on each plate
    proportion_walking_events_p1 = walking_events_p1 / total_walking_events if total_walking_events > 0 else 0
    proportion_walking_events_p2 = walking_events_p2 / total_walking_events if total_walking_events > 0 else 0

    # Calculate the difference in proportion of walking events between the two plates
    walking_event_difference = proportion_walking_events_p2 - proportion_walking_events_p1

    # Calculate the normalized time metric
    normalized_time_metric = (corrected_average_time_2 - corrected_average_time_1) / (corrected_average_time_1 + corrected_average_time_2)

    # Calculate the normalized distance metric
    normalized_distance_metric = (average_distance_on_plate2 - average_distance_on_plate1) / (average_distance_on_plate1 + average_distance_on_plate2)

    # Calculate the Host Seeking Index (HSI) for each plate
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
        "DTT Score": [round(dtt_score, 4)],
    }




    df = pd.DataFrame.from_dict(results)


    df.to_csv(outpath, mode="a", index=False, header=False)


    print("Results saved to CSV file:", outpath)


if __name__ == '__main__':
    main()