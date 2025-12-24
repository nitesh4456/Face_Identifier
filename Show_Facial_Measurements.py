import face_recognition
import cv2
import os
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import Tk, filedialog
from itertools import permutations
import matplotlib.pyplot as plt
import cv2
import numpy as np
# Renamed center function to center1
def center1(landmarks):
    # Calculate the center of a set of landmarks by averaging the x and y coordinates
    x = np.mean([p[0] for p in landmarks])
    y = np.mean([p[1] for p in landmarks])
    return int(x), int(y)  # Ensure the center is returned as an integer tuple
def draw_attribute_lines(image_cv, face_landmarks, attr):
    # Colors for each attribute (feel free to adjust)
    def draw_round_feature(feature_name, color):
        if feature_name in face_landmarks:
            points = np.array(face_landmarks[feature_name], dtype=np.int32)
            points = points.reshape((-1, 1, 2))  # Convert to required shape for polylines
            cv2.polylines(image_cv, [points], isClosed=True, color=color, thickness=2)# Draw a closed line for round-shaped features using cv2.polylines
    def color_assigner(attr_name):
        # return (0, 255, 255)  # Default color 
        color_map= {
        "inter_eye_distance": (255, 0, 0),  # Blue
        "left_eye_width": (0, 255, 0),     # Green
        "right_eye_width": (0, 0, 255),    # Red
        "left_eye_height": (255, 255, 0),  # Yellow
        "right_eye_height": (0, 255, 255), # Cyan
        "left_eye_aspect_ratio": (255, 0, 255),  # Magenta
        "right_eye_aspect_ratio": (255, 165, 0), # Orange
        "eye_alignment_angle": (128, 0, 128),  # Purple
        "nose_bridge_length": (0, 128, 0),  # Dark Green
        "nose_width": (255, 0, 255),       # Pink
        "nose_height": (128, 128, 0),      # Olive
        "left_eye_to_nose_tip": (255, 105, 180), # Hot Pink
        "right_eye_to_nose_tip": (144, 238, 144), # Light Green
        "nose_tip_to_mouth": (0, 139, 139), # Dark Cyan
        "mouth_width": (0, 0, 139),        # Dark Blue
        "mouth_height": (255, 20, 147),    # Deep Pink
        "top_lip_thickness": (0, 255, 127), # Spring Green
        "bottom_lip_thickness": (0, 191, 255), # Deep Sky Blue
        "mouth_aspect_ratio": (255, 69, 0), # Red Orange
        "nose_to_mouth_distance": (138, 43, 226), # Blue Violet
        "lip_curve_angle": (0, 255, 255),   # Yellow
        "eye_width_diff": (255, 105, 180), # Hot Pink
        "eyebrow_height_diff": (255, 255, 255), # White
        "chin_symmetry": (255, 20, 147),   # Deep Pink
        "eye_nose_symmetry": (255, 255, 0), # Yellow
        "jaw_width": (0, 139, 139),        # Dark Cyan
        "face_height": (255, 99, 71),      # Tomato
        "face_ratio": (72, 61, 139),       # Dark Slate Blue
        "chin_roundness": (0, 255, 255),   # Yellow
        "avg_eyebrow_slope": (123, 104, 238) # Medium Slate Blue
        }
        # Assign colors based on the attribute name
        if attr_name in color_map:
            return color_map[attr_name]
        else:
            return (255, 255, 255)
    

    # Define the line drawing function for each specific attribute
    def draw_line(p1, p2, color):
        if p1 and p2:  # Check if points are valid
            cv2.line(image_cv, p1, p2, color, 2)  # Draw line with thickness = 2

    # Draw lines based on the attributes
    if "left_eye" in face_landmarks and "right_eye" in face_landmarks:
        # Inter-eye distance
        draw_line(center1(face_landmarks["left_eye"]), center1(face_landmarks["right_eye"]), color_assigner("inter_eye_distance"))

        # Eye widths (left and right)
        draw_line(face_landmarks["left_eye"][0], face_landmarks["left_eye"][3], color_assigner("left_eye_width"))
        draw_line(face_landmarks["right_eye"][0], face_landmarks["right_eye"][3], color_assigner("right_eye_width"))

        # Eye heights
        draw_line(face_landmarks["left_eye"][1], face_landmarks["left_eye"][5], color_assigner("left_eye_height"))
        draw_line(face_landmarks["left_eye"][2], face_landmarks["left_eye"][4], color_assigner("left_eye_height"))
        draw_line(face_landmarks["right_eye"][1], face_landmarks["right_eye"][5], color_assigner("right_eye_height"))
        draw_line(face_landmarks["right_eye"][2], face_landmarks["right_eye"][4], color_assigner("right_eye_height"))

    if "nose_bridge" in face_landmarks and "nose_tip" in face_landmarks:
        # Nose bridge length
        draw_line(face_landmarks["nose_bridge"][0], face_landmarks["nose_bridge"][-1], color_assigner("nose_bridge_length"))

        # Nose width and height
        draw_line(face_landmarks["nose_tip"][0], face_landmarks["nose_tip"][-1], color_assigner("nose_width"))
        draw_line(face_landmarks["nose_bridge"][0], center1(face_landmarks["nose_tip"]), color_assigner("nose_height"))

        # Eye to nose tip distances
        draw_line(center1(face_landmarks["left_eye"]), center1(face_landmarks["nose_tip"]), color_assigner("left_eye_to_nose_tip"))
        draw_line(center1(face_landmarks["right_eye"]), center1(face_landmarks["nose_tip"]), color_assigner("right_eye_to_nose_tip"))

    if "top_lip" in face_landmarks and "bottom_lip" in face_landmarks:
        # Mouth width and height
        draw_line(face_landmarks["top_lip"][0], face_landmarks["top_lip"][6], color_assigner("mouth_width"))
        draw_line(face_landmarks["top_lip"][6], face_landmarks["bottom_lip"][6], color_assigner("mouth_height"))

        # Lip thickness
        draw_line(face_landmarks["top_lip"][3], face_landmarks["top_lip"][9], color_assigner("top_lip_thickness"))
        draw_line(face_landmarks["bottom_lip"][3], face_landmarks["bottom_lip"][9], color_assigner("bottom_lip_thickness"))

    if "chin" in face_landmarks:
        # Chin symmetry and jaw width
        draw_line(face_landmarks["chin"][0], face_landmarks["chin"][-1], color_assigner("jaw_width"))

    # Return the image with the drawn lines
    # Draw the round features by calling the helper function
    draw_round_feature("chin", color_assigner("chin_roundness"))
    draw_round_feature("top_lip", color_assigner("top_lip_thickness"))
    draw_round_feature("bottom_lip", color_assigner("bottom_lip_thickness"))
    draw_round_feature("mouth", color_assigner("mouth_height"))
    draw_round_feature("left_eyebrow", color_assigner("left_eye_width"))
    draw_round_feature("right_eyebrow", color_assigner("right_eye_width"))
    return image_cv
def enhance_to_square(points):
    """
    Given 4 points (x, y), returns 4 new points forming a square
    such that all pairwise distances are increased or unchanged.
    """
    points = np.array(points)

    # Find axis-aligned bounding box
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    width = max_x - min_x
    height = max_y - min_y

    # Expand to square
    side = max(width, height)

    # Make the square start from (min_x, min_y)
    # Adjust max_x or max_y if needed to make it a square
    if width < height:
        max_x = min_x + side
    else:
        max_y = min_y + side

    # Define square corners (counter-clockwise)
    square_corners = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])

    # Find the assignment of square corners to points that maximizes total distance
    best_match = None
    max_total_dist = -np.inf

    for perm in permutations(square_corners):
        perm = np.array(perm)
        dists = np.linalg.norm(perm - points, axis=1)
        orig_dists = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        new_dists = np.linalg.norm(perm[:, None] - perm[None, :], axis=-1)
        # Check if all distances are increased or unchanged
        if np.all(new_dists >= orig_dists):
            total_dist = dists.sum()
            if total_dist > max_total_dist:
                max_total_dist = total_dist
                best_match = perm

    if best_match is not None:
        return best_match.tolist()
    else:
        raise ValueError("No valid square transformation found without reducing distances.")

def order_square_points(points):
    # points: list of [x, y] pairs
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required.")

    # Sort points by Y (top to bottom), then by X (left to right)
    points = sorted(points, key=lambda p: (p[1], p[0]))  # sort by y, then x

    top_two = sorted(points[:2], key=lambda p: p[0])   # sort top points left to right
    bottom_two = sorted(points[2:], key=lambda p: p[0])  # sort bottom points left to right

    # Clockwise: top-left, top-right, bottom-right, bottom-left
    return [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]

def dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def center(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

def angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def main():
    # Open file dialog to select image
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    file_name = os.path.basename(file_path)
    results = []
    name="unknown"
    # Load the selected image
    image = face_recognition.load_image_file(file_path)
    height, width = image.shape[:2]
    print(f"Original Image Size: Width = {width}, Height = {height}")
    diag1 = math.hypot(width, height)
    print(f"Image Diagonal Size: {diag1}")
    # Find all face landmarks in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    if not face_landmarks_list:
        print(f"No face found in: {file_name}")
        quit()
    if face_landmarks_list:
        for face_landmarks in face_landmarks_list:
            left_eye = face_landmarks["left_eye"]
            right_eye = face_landmarks["right_eye"]
            nose_bridge = face_landmarks["nose_bridge"]
            nose_tip = face_landmarks["nose_tip"]
            top_lip = face_landmarks["top_lip"]
            bottom_lip = face_landmarks["bottom_lip"]
            chin = face_landmarks["chin"]
            left_eyebrow = face_landmarks["left_eyebrow"]
            right_eyebrow = face_landmarks["right_eyebrow"]

            ######################################################Exact diagonal calculation by having the square of the face not image###############

            #defining the coordinate 1, 2, 3 , 4 for top most , bottom most , left most and right most
            coordinate1=left_eyebrow[0]
            for ele in left_eyebrow:
                if ele[1]<coordinate1[1]:#smaller means more top
                    coordinate1=ele
            for ele in right_eyebrow:
                if ele[1]<coordinate1[1]:
                    coordinate1=ele
            #now i have a face height top coordinate value
            coordinate2=chin[9]
            for i in range(6, 13):
                if chin[i][1]>coordinate2[1]:
                    coordinate2=chin[i]
            #now i have a face height bottom value
            coordinate3 = left_eyebrow[0]
            for ele in left_eyebrow:
                if ele[0]<coordinate3[0]:
                    coordinate3=ele
            for i in range(0, 9):
                if chin[i][0]<coordinate3[0]:
                    coordinate3=chin[i]

            coordinate4 = right_eyebrow[0]
            for ele in right_eyebrow:
                if ele[0]>coordinate4[0]:
                    coordinate4=ele
            for i in range(9, 17):
                if chin[i][0]>coordinate4[0]:
                    coordinate4=chin[i]

            # print("coordinate1",coordinate1)
            # print("coordinate2",coordinate2)
            # print("coordinate3",coordinate3)
            # print("coordinate4",coordinate4)

            original_points = [coordinate1, coordinate2, coordinate3, coordinate4]
            try:
                square_points = enhance_to_square(original_points)
            except ValueError as e:
                print(f"Error enhancing to square for {file_name}: {e}")
                quit()

            # square_points_int = [(int(x), int(y)) for x, y in square_points]
            ordered = order_square_points(square_points)


            ######################################################Exact diagonal calculation by having the square of the face not image###############
            #######Exact visualization of the face landmarks###############
            # You can visualize the landmarks on the image if you want
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            plt.imshow(image_cv)
            plt.show()
            p=input("Press any key to continue")
            #DRAWING THE SQUARE ON THE IMAGE
            for i in range(4):
                start = ordered[i]
                end = ordered[(i + 1) % 4]  # Loop around to first point
                cv2.line(image_cv, start, end, (0, 255, 0), 2)
            plt.imshow(image_cv)
            plt.show()
            print("Square coordinates:")#rember the square points are calculated using coordinates points and the face landmarks
            for pt in ordered:
                print(pt)# Clockwise: top-left, top-right, bottom-right, bottom-left
            diag=math.hypot(dist(ordered[0], ordered[1]), dist(ordered[0], ordered[3]))
            print("Diagonal of the square:", diag)

            p=input("Press any key to continue")
            # Draw landmarks using blue dots
            for facial_feature in face_landmarks:
                for (x, y) in face_landmarks[facial_feature]:
                    cv2.circle(image_cv, (x, y), 1, (255, 0, 0), -1)
            # Show the image with landmarks
            plt.imshow(image_cv)
            plt.show()





            #######Exact visualization of the face landmarks###############
            attr = {

                "inter_eye_distance": dist(center(left_eye), center(right_eye))/diag,
                "left_eye_width": dist(left_eye[0], left_eye[3])/diag,
                "right_eye_width": dist(right_eye[0], right_eye[3])/diag,
                "left_eye_height": ((dist(left_eye[1], left_eye[5]) + dist(left_eye[2], left_eye[4])) / 2)/diag,
                "right_eye_height": ((dist(right_eye[1], right_eye[5]) + dist(right_eye[2], right_eye[4])) / 2)/diag,
                "left_eye_aspect_ratio": ((dist(left_eye[1], left_eye[5]) + dist(left_eye[2], left_eye[4])) / 2) / dist(left_eye[0], left_eye[3]),
                "right_eye_aspect_ratio": ((dist(right_eye[1], right_eye[5]) + dist(right_eye[2], right_eye[4])) / 2) / dist(right_eye[0], right_eye[3]),
                "eye_alignment_angle": angle(left_eye[0], right_eye[3]),
                "nose_bridge_length": dist(nose_bridge[0], nose_bridge[-1])/diag,
                "nose_width": dist(nose_tip[0], nose_tip[-1])/diag,
                "nose_height": dist(nose_bridge[0], center(nose_tip))/diag,
                "left_eye_to_nose_tip": dist(center(left_eye), center(nose_tip))/diag,
                "right_eye_to_nose_tip": dist(center(right_eye), center(nose_tip))/diag,
                "nose_tip_to_mouth": dist(center(nose_tip), top_lip[6])/diag,
                "mouth_width": dist(top_lip[0], top_lip[6])/diag,
                "mouth_height": dist(top_lip[6], bottom_lip[6])/diag,
                "top_lip_thickness": dist(top_lip[3], top_lip[9])/diag,
                "bottom_lip_thickness": dist(bottom_lip[3], bottom_lip[9])/diag,
                "mouth_aspect_ratio": dist(top_lip[6], bottom_lip[6]) / dist(top_lip[0], top_lip[6]),
                "nose_to_mouth_distance": dist(nose_bridge[-1], top_lip[6])/diag,
                "lip_curve_angle": angle(top_lip[0], top_lip[-1]),
                "eye_width_diff": abs(dist(left_eye[0], left_eye[3]) - dist(right_eye[0], right_eye[3]))/diag,
                "eyebrow_height_diff": abs(left_eyebrow[2][1] - right_eyebrow[2][1])/diag,
                "chin_symmetry": abs(dist(chin[0], chin[8]) - dist(chin[-1], chin[8]))/diag,
                "eye_nose_symmetry": abs(dist(left_eye[3], nose_tip[0]) - dist(right_eye[0], nose_tip[-1]))/diag,
                "jaw_width": dist(chin[0], chin[-1])/diag,
                "face_height": dist(nose_bridge[0], chin[8])/diag,
                "face_ratio": dist(chin[0], chin[-1]) / dist(nose_bridge[0], chin[8]),
                "chin_roundness": np.mean([dist(chin[i], chin[8]) for i in [3, 4, 12, 13]])/diag,
                "avg_eyebrow_slope": np.mean([
                    angle(left_eyebrow[0], left_eyebrow[-1]),
                    angle(right_eyebrow[0], right_eyebrow[-1])]),
                "Image": name
            }

            results.append(attr)


            # Draw lines for attributes
            p=input("Press any key to continue")
            image_with_lines = draw_attribute_lines(image_cv, face_landmarks,{})
            plt.imshow(image_with_lines)
            plt.show()

main()