import tfjs_graph_converter as tfjs
import tensorflow as tf
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


# PATHS
imagePath = 'path/to/.jpg/file'
modelPath = 'path/to/folder/containing/model.json'

# CONSTANTS
OutputStride = 16

KEYPOINT_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]


KEYPOINT_IDS = {name: id for id, name in enumerate(KEYPOINT_NAMES)}

CONNECTED_KEYPOINTS_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_KEYPOINT_INDICES = [(KEYPOINT_IDS[a], KEYPOINT_IDS[b])
                              for a, b in CONNECTED_KEYPOINTS_NAMES]


POSE_CHAIN = [
    ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
    ("rightEye", "rightEar"), ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle")
]

PARENT_CHILD_TUPLES = [(KEYPOINT_IDS[parent], KEYPOINT_IDS[child])
                       for parent, child in POSE_CHAIN]


print("Loading model...", end="")
graph = tfjs.api.load_graph_model(modelPath)  # downloaded from the link above
print("done.\nLoading sample image...", end="")


def getBoundingBox(keypointPositions, offset=(10, 10, 10, 10)):
    minX = math.inf
    minY = math.inf
    maxX = - math.inf
    maxY = -math.inf
    for y, x in keypointPositions:
        if (x < minX):
            minX = x
        if(y < minY):
            minY = y
        if(x > maxX):
            maxX = x
        if (y > maxY):
            maxY = y
    return (minX - offset[0], minY-offset[1]), (maxX+offset[2], maxY + offset[3])

# Find Displacement by traversing to target keypoint


def traverse_to_targ_keypoint(edge_id, source_keypoint, target_keypoint_id, scores, offsets, output_stride, displacements):
    height = scores.shape[0]
    width = scores.shape[1]

    source_keypoint_indices = np.clip(
        np.round(source_keypoint / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    displaced_point = source_keypoint + displacements[
        source_keypoint_indices[0], source_keypoint_indices[1], edge_id]

    displaced_point_indices = np.clip(
        np.round(displaced_point / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    score = scores[displaced_point_indices[0],
                   displaced_point_indices[1], target_keypoint_id]

    image_coord = displaced_point_indices * output_stride + offsets[
        displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    return score, image_coord


def get_instance_score_fast(
        exist_pose_coords,
        squared_nms_radius,
        keypoint_scores, keypoint_coords):

    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords)
                   ** 2, axis=2) > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)


def within_nms_radius_fast(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)


# load sample image into numpy array
img = tf.keras.preprocessing.image.load_img(imagePath)
imgWidth, imgHeight = img.size

targetWidth = (int(imgWidth) // OutputStride) * OutputStride + 1
targetHeight = (int(imgHeight) // OutputStride) * OutputStride + 1

print(imgHeight, imgWidth, targetHeight, targetWidth)
img = img.resize((targetWidth, targetHeight))
x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
InputImageShape = x.shape
print("Input Image Shape in hwc", InputImageShape)


widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1
print('Resolution', widthResolution, heightResolution)

# add imagenet mean - extracted from body-pix source
m = np.array([-123.15, -115.90, -103.06])
x = np.add(x, m)
sample_image = x[tf.newaxis, ...]
print("done.\nRunning inference...", end="")

# evaluate the loaded model directly
with tf.compat.v1.Session(graph=graph) as sess:
    input_tensor_names = tfjs.util.get_input_tensors(graph)
    print(input_tensor_names)
    output_tensor_names = tfjs.util.get_output_tensors(graph)
    print(output_tensor_names)
    input_tensor = graph.get_tensor_by_name(input_tensor_names[0])
    results = sess.run(output_tensor_names, feed_dict={
                       input_tensor: sample_image})
print("done. {} outputs received".format(len(results)))  # should be 4 outputs
displacements_fwd = np.squeeze(results[0], 0)
print('fwd', displacements_fwd.shape)
displacements_bwd = np.squeeze(results[1], 0)
print('bwd', displacements_bwd.shape)
offsets = np.squeeze(results[2], 0)
print('offsets', offsets.shape)
heatmaps = np.squeeze(results[3], 0)
print('heatmaps', heatmaps.shape)


# ##########
# Mutipose Estimation
# ##########

# Constants
max_pose_detections = 10
NUM_KEYPOINTS = len(KEYPOINT_NAMES)
LOCAL_MAXIMUM_RADIUS = 1
nms_radius = 20
score_threshold = 0.5
min_pose_score = 0.5

# Initialize pose data
pose_count = 0
pose_scores = np.zeros(max_pose_detections)
pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

squared_nms_radius = nms_radius ** 2

height = heatmaps.shape[0]
width = heatmaps.shape[1]

scored_parts = []
for hmy in range(height):
    for hmx in range(width):
        for keypoint_id in range(heatmaps.shape[2]):
            score = heatmaps[hmy, hmx, keypoint_id]
            if score < score_threshold:
                continue

            y_start = max(hmy - LOCAL_MAXIMUM_RADIUS, 0)
            y_end = min(hmy + LOCAL_MAXIMUM_RADIUS + 1, height)
            x_start = max(hmx - LOCAL_MAXIMUM_RADIUS, 0)
            x_end = min(hmx + LOCAL_MAXIMUM_RADIUS + 1, width)

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if heatmaps[y, x, keypoint_id] <= score:
                        scored_parts.append((
                            score, keypoint_id, np.array((hmy, hmx))
                        ))
# Sort the parts by descending score
scored_parts = sorted(scored_parts, key=lambda x: x[0], reverse=True)

offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
displacements_fwd = displacements_fwd.reshape(
    height, width, 2, -1).swapaxes(2, 3)
displacements_bwd = displacements_bwd.reshape(
    height, width, 2, -1).swapaxes(2, 3)

# Use the keypoint confidence score, keypointid, keypoint position
for root_score, root_id, root_coord in scored_parts:
    # Find original position in image: position * outputStride + offset
    root_image_coords = root_coord * OutputStride + \
        offsets[root_coord[0], root_coord[1], root_id]

    # Check if within nms radius
    if within_nms_radius_fast(
            pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords):
        continue

    num_edges = len(PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(NUM_KEYPOINTS)
    instance_keypoint_coords = np.zeros((NUM_KEYPOINTS, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coords

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                heatmaps, offsets, OutputStride, displacements_bwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords
    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                heatmaps, offsets, OutputStride, displacements_fwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    pose_score = get_instance_score_fast(
        pose_keypoint_coords[:pose_count, :, :], squared_nms_radius, instance_keypoint_scores, instance_keypoint_coords)

    # NOTE this isn't in the original implementation, but it appears that by initially ordering by
    # part scores, and having a max # of detections, we can end up populating the returned poses with
    # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
    # Set min_pose_score to 0. to revert to original behaviour
    if min_pose_score == 0. or pose_score >= min_pose_score:
        pose_scores[pose_count] = pose_score
        pose_keypoint_scores[pose_count, :] = instance_keypoint_scores
        pose_keypoint_coords[pose_count, :, :] = instance_keypoint_coords
        pose_count += 1

    if pose_count >= max_pose_detections:
        break

# RESULTS are now in pose_scores, pose_keypoint_scores, pose_keypoint_coords
for posenum, pose_score in enumerate(pose_scores):
    if pose_score > min_pose_score:
        print('Pose Scores', pose_score)
        print('Pose Keypoint Scores', pose_keypoint_scores[posenum])
        print('Pose Keypoint Coords', pose_keypoint_coords[posenum])

        # Show all Keypoints
        implot = plt.imshow(img)
        x_points = []
        y_points = []
        for y, x in pose_keypoint_coords[posenum]:
            x_points.append(x)
            y_points.append(y)
        plt.scatter(x=x_points, y=y_points, c='r', s=40)
        plt.show()

        # Show Connected Keypoints
        plt.figure(20)
        for pt1, pt2 in CONNECTED_KEYPOINT_INDICES:
            plt.title('connection points')
            implot = plt.imshow(img)
            plt.plot((pose_keypoint_coords[posenum][pt1][1], pose_keypoint_coords[posenum][pt2][1]), (
                pose_keypoint_coords[posenum][pt1][0], pose_keypoint_coords[posenum][pt2][0]), 'ro-', linewidth=2, markersize=5)
        plt.show()

        # Get Bounding BOX
        (xmin, ymin), (xmax, ymax) = getBoundingBox(
            pose_keypoint_coords[posenum])
        print('Bonding Box xmin,ymin, xmax, ymax format: ', xmin, ymin, xmax, ymax)

        # Show Bounding BOX
        implot = plt.imshow(img)  # Get the current reference / axis
        ax = plt.gca()  # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                 linewidth=1, edgecolor='r', facecolor='none', fill=False)

        ax.add_patch(rect)  # Add the patch
        plt.show()
