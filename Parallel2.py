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
import os


def main(video_folder, out_video_dir):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mov', '.avi', '.mp4'))]
    
    # Initialize roi_points
    roi_points = []

    # Select ROIs using the first video
    if video_files:
        first_video_path = os.path.join(video_folder, video_files[0])
        video = cv2.VideoCapture(first_video_path)  # Add this line
        select_rois(first_video_path, roi_points)

    # Process all videos using the selected ROIs
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        process_video(video_path, out_video_dir, roi_points)



def select_rois(video_path, roi_points):
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    frame_copy = frame.copy()
    cv2.namedWindow('Select Regions of Interest')
    
    # Using functools.partial
    from functools import partial
    cv2.setMouseCallback('Select Regions of Interest', partial(select_roi, roi_points=roi_points, frame=frame_copy))
    
    while True:
        cv2.imshow('Select Regions of Interest', frame_copy)
        key = cv2.waitKey(1) & 0xFF

        # Break the loop when 'q' key is pressed
        if key == ord("q"):
            break


cv2.destroyAllWindows()

def process_video(video_path, out_video_dir, roi_points):
    video_name = os.path.basename(video_path)
    video = cv2.VideoCapture(video_path)
    out_video_name = video_name.split('.')[0] + '_output.avi'
    out_video_path = os.path.join(out_video_dir, out_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print("Video FPS: ", fps)


    def preprocess_frame(frame):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        return blurred_frame





def select_roi(event, x, y, flags, param, frame, roi_points):
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        cv2.rectangle(frame, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
        cv2.imshow('Select Regions of Interest', frame)


    # This function detects mosquitoes in the frame and returns their bounding box coordinates.
    # It takes a frame, a list of regions of interest (ROIs), and a minimum area value.


    def detect_mosquitoes(frame, rois, min_area=1):
        # Apply the background subtraction to the frame.
        fgmask = fgbg.apply(frame)
        # Threshold the foreground mask to create a binary image.
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        # Find contours in the binary image.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize an empty list to store mosquitoes.
        mosquitoes = []

        # Loop through each contour.
        for contour in contours:
            # If the contour area is greater than the minimum area, it's considered a mosquito.
            if cv2.contourArea(contour) > min_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                mosquito_center = (x + w // 2, y + h // 2)

                # Check if the mosquito center is within any of the regions of interest (ROIs).
                for roi in rois:
                    if (roi[0][0] <= mosquito_center[0] <= roi[1][0]) and (roi[0][1] <= mosquito_center[1] <= roi[1][1]):
                        mosquitoes.append((x, y, w, h))
                        break

        # Return the list of detected mosquitoes.
        return mosquitoes

    fps = 10
    # Calculate the state change capacity based on 80% of the video's FPS.
    state_cap = int(fps * .8)


    # MosquitoTracker class for tracking individual mosquitoes.
    class MosquitoTracker:
        # Initialize the MosquitoTracker with a maximum number of frames.
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


        # Update the tracker's position and area with a new position.
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


        # Determine if the mosquito is moving based on a given threshold.
        def is_moving(self, threshold=5):
            if len(self.positions) < 2:
                return False

            # Calculate the distance between consecutive positions.
            for i in range(len(self.positions) - 1):
                dist = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[i + 1]))
                # If the distance is greater than the threshold, the mosquito is considered moving.
                if dist > threshold:
                    self.host_seeking_count += 1
                    return True
            return False


    # This function assigns mosquitoes to their corresponding trackers.
    def track_mosquitoes(mosquitoes, trackers, rois):
        # Loop through each detected mosquito.
        for mosquito in mosquitoes:
            x, y, w, h = mosquito
            center = (x + w // 2, y + h // 2)

            # Initialize the minimum distance and closest tracker variables.
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
    # Loop through each tracker.
            for tracker_id, tracker in trackers.items():
                # Calculate the distance between the tracker's last position and the mosquito center.
                distance = np.linalg.norm(np.array(tracker.positions[-1]) - np.array(center))
                # Update the minimum distance and closest tracker if the distance is smaller.
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
        frame_copy = frame.copy()
        cv2.namedWindow('Select Regions of Interest')
        cv2.setMouseCallback('Select Regions of Interest', lambda *args: select_roi(*args, roi_points))

        while True:
            cv2.imshow('Select Regions of Interest', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break

        cv2.destroyWindow('Select Regions of Interest')

    rois = [(roi_points[i], roi_points[i + 1]) for i in range(0, len(roi_points), 2)]


    frame_height, frame_width, _ = frame.shape
    heatmap_p1 = np.zeros((frame_height, frame_width), dtype=np.uint32)
    heatmap_p2 = np.zeros((frame_height, frame_width), dtype=np.uint32)
    trace_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Create the video writer object
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
                    color = (0, 255, 0)  # Green for walking paths
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
                color = (0, 0, 255)  # Red for moving mosquitoes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if (rois[0][0][0] <= x + w // 2 <= rois[0][1][0]) and (rois[0][0][1] <= y + h // 2 <= rois[0][1][1]):
                    heatmap_p1[y:y+h, x:x+w] += 1
                elif (rois[1][0][0] <= x + w // 2 <= rois[1][1][0]) and (rois[1][0][1] <= y + h // 2 <= rois[1][1][1]):
                    heatmap_p2[y:y+h, x:x+w] += 1

                # Check if the mosquito is in P1
                if (rois[0][0][0] <= x + w // 2 <= rois[0][1][0]) and (rois[0][0][1] <= y + h // 2 <= rois[0][1][1]):
                    walking_events_p1 += 1
                # Check if the mosquito is in P2
                elif (rois[1][0][0] <= x + w // 2 <= rois[1][1][0]) and (rois[1][0][1] <= y + h // 2 <= rois[1][1][1]):
                    walking_events_p2 += 1



        for roi in rois:
            cv2.rectangle(frame, roi[0], roi[1], (255, 255, 0), 2)

        out_video.write(frame)

        # Display the frame
        cv2.imshow('Mosquito Tracker', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    out_video.release()
    cv2.destroyAllWindows()

    bin_size = 8

    #calculate bin count which is number of frames divided by the stated bin size
    bin_count = frame_count // bin_size


    if total_walking_events > 0:
        proportion_p1 = walking_events_p1 / total_walking_events
        proportion_p2 = walking_events_p2 / total_walking_events
        hpi = proportion_p2 - proportion_p1
    else:
        hpi = 0

    hsi = total_walking_events / frame_count

    # Initialize counters for total time on each plate
    total_time_on_plate1 = 0
    total_time_on_plate2 = 0

    # Calculate the total time spent on each plate by all mosquitoes
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

    # print("Average distance on plate 1:", round(average_distance_on_plate1,3))
    # print("Average distance on plate 2:", round(average_distance_on_plate2,3))


    # Calculate the average time spent on each plate
    average_time_on_plate1 = total_time_on_plate1 / len(trackers)
    average_time_on_plate2 = total_time_on_plate2 / len(trackers)

    corrected_average_time_1 = average_time_on_plate1/fps
    corrected_average_time_2 = average_time_on_plate2/fps

    normalized_time_metric = corrected_average_time_2 / (corrected_average_time_1 + corrected_average_time_2)
    normalized_distance_metric = average_distance_on_plate2 / (average_distance_on_plate1 + average_distance_on_plate2)

    # print(corrected_average_time_1)
    # print(corrected_average_time_2)


    #calculate HSI for each plate
    hsi_p1 = walking_events_p1 / frame_count
    hsi_p2 = walking_events_p2 / frame_count

    # Normalize the heatmaps
    heatmap_p1_normalized = heatmap_p1 / np.max(heatmap_p1)
    heatmap_p2_normalized = heatmap_p2 / np.max(heatmap_p2)


    combined_heatmap = heatmap_p1_normalized + heatmap_p2_normalized

    # Normalize the combined heatmap
    combined_heatmap_normalized = combined_heatmap / np.max(combined_heatmap)


    data = {
        "HPI": [round(hpi, 4)],
        "Total Walking Events": [total_walking_events],
        "Walks Cont Plate": [walking_events_p1],
        "Walks IR Plate": [walking_events_p2],
        "HSI": [round(hsi, 4)],
        "HSI Control Plate": [round(hsi_p1, 4)],
        "HSI IR Plate": [round(hsi_p2, 4)],
        "Normalized Avg Time Metric": [round(normalized_time_metric, 4)],
        "Normalized Avg Distance Metric": [round(normalized_distance_metric, 4)]
    }

    table_data = pd.DataFrame(data)

    outpath = "//cmontell-nas1.mcdb.ucsb.edu/Ryan/Mosqutio Tracker 2 way choice/Tracker_2Plate_Results.csv"
    #outpath= "/Users/ryandonofrio/Desktop/Tracker/Results.csv"

    # Check if the results.csv file exists, and create it if it doesn't
    if not os.path.exists(outpath):
        with open(outpath, "w") as f:
            f.write("Video Name,HPI,Total Walk events,Walk Cont, Walk IR, HSI, HSI Cont, HSI IR, Normalized Time, Normalized Distance, Tot Time P1, Tot Time P2, Tot Distance P1, Tot Distance P2\n")

    # Create a dictionary to store the results
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
        "Total Time P2": [round(total_time_on_plate1, 4)],
        "Total Distance P1": [round(total_distance_on_plate1, 4)],
        "Total Distance P2": [round(total_distance_on_plate2, 4)],

    }



    # Create a pandas DataFrame from the results dictionary
    df = pd.DataFrame.from_dict(results)

    # Append the DataFrame to the CSV file
    df.to_csv(outpath, mode="a", index=False, header=False)

    # Print a message to confirm that the results were saved
    print("Results saved to CSV file:", outpath)


if __name__ == "__main__":
    video_folder = r"\\cmontell-nas1.mcdb.ucsb.edu\Ryan\Mosqutio Tracker 2 way choice\Parallel Processing\In Test 1"
    out_video_dir = r"\\cmontell-nas1.mcdb.ucsb.edu\Ryan\Mosqutio Tracker 2 way choice\Parallel Processing\Out Test 1"
    main(video_folder, out_video_dir)
