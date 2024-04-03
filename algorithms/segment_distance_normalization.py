import numpy as np
import copy

import config 

skeleton_graph = dict({
    0 : [2, 7],
    1 : [2, 8],
    2 : [0, 1, 3, 19],
    3 : [2, 6],
    4 : [6, 14],
    5 : [6, 13],
    6 : [3, 4, 5],
    7 : [0, 9],
    8 : [1, 10],
    9 : [7, 11],
    10: [8, 12],
    11: [9],
    12: [10],
    13: [5, 15],
    14: [4, 16],
    15: [13, 17],
    16: [14, 18],
    17: [15],
    18: [16],
    19: [2],
})

avg_dist = None

edges = []
for u, neighbours in skeleton_graph.items():
  for v in neighbours:
    edges.append((u, v))
edges = list(set(edges))
assert len(edges) == 2*(config.NUM_JOINTS-1)

skeleton_edges = [
    (3, 2), (2, 0), (0, 7), (7, 9), (9, 11), (2, 1), (1, 8), (8, 10), (10, 12),
    (2, 19), (3, 6), (6, 5), (5, 13), (13, 15), (15, 17), (6, 4), (4, 14),
    (14, 16), (16, 18)
]
assert len(skeleton_edges) == (config.NUM_JOINTS - 1)

def get_distance(point1, point2):
  assert point1.shape == (3,)
  assert point2.shape == (3,)
  return np.linalg.norm(point1 - point2)

def accumulate_distance(frame, u, v, avg_dist):
  if not (u>=0 and u<config.NUM_JOINTS and v>=0 and v<config.NUM_JOINTS):
    return
  assert (u, v) in skeleton_edges
  d = get_distance(frame[u], frame[v])
  if d==0:
    raise ValueError
  avg_dist[u][v]+=d

def accumulate_frame_distances(frame, avg_dist):
  assert frame.shape == (config.NUM_JOINTS, 3)
  start = config.ROOT_JOINT_INDEX
  visited, q, par = [], [start], [-1]
  seen_edges = []
  while q:
    u = q.pop(0)
    p = par.pop(0)
    if u not in visited:
      visited.append(u)
      accumulate_distance(frame, p, u, avg_dist)
      seen_edges.append((p, u))
      for v in skeleton_graph[u]:
        if v not in visited:
          q.append(v)
          par.append(u)
  assert len(seen_edges) == config.NUM_JOINTS
  assert len(list(set(seen_edges))) == len(seen_edges)

def get_joint_segment_avg_distances(data):
  avg_dist = np.zeros((config.NUM_JOINTS, config.NUM_JOINTS))
  num_frames = 0
  for sequence in data:
    for frame in sequence:
      num_frames+=1
      accumulate_frame_distances(frame, avg_dist)
  avg_dist/=num_frames
  for i in range(config.NUM_JOINTS):
    for j in range(config.NUM_JOINTS):
      if (i, j) not in skeleton_edges:
        assert avg_dist[i][j] == 0
      else:
        print(i, j, avg_dist[i][j])
  return avg_dist


def get_avg_dist(data_use):
  global avg_dist
  avg_dist = get_joint_segment_avg_distances(data_use)


def distance_normalize_segment(frame, u, v, modified_frame):
  global avg_dist
  if not (u>=0 and u<config.NUM_JOINTS and v>=0 and v<config.NUM_JOINTS):
    return
  assert (u, v) in skeleton_edges
  segment = frame[v] - frame[u]
  assert np.linalg.norm(segment) >= 1e-5
  modified_frame[v] = modified_frame[u] + avg_dist[u][v]*(segment/(np.linalg.norm(segment)))

def distance_normalize_frame(frame):
  assert frame.shape == (config.NUM_JOINTS, 3)
  assert np.array_equal(frame[3], np.array([0, 0, 0]))
  modified_frame = copy.deepcopy(frame)
  start = config.ROOT_JOINT_INDEX
  visited, q, par = [], [start], [-1]
  seen_edges = []
  while q:
    u = q.pop(0)
    p = par.pop(0)
    if u not in visited:
      visited.append(u)
      distance_normalize_segment(frame, p, u, modified_frame)
      seen_edges.append((p, u))
      for v in skeleton_graph[u]:
        if v not in visited:
          q.append(v)
          par.append(u)
  assert len(seen_edges) == config.NUM_JOINTS
  assert len(list(set(seen_edges))) == len(seen_edges)
  return modified_frame

def apply_segment_distance_normalization(sequence):
  if not config.APPLY_SEGMENT_DISTANCE_NORMALIZATION:
    raise ValueError
  for frame_ind in range(sequence.shape[0]):
    sequence[frame_ind] = distance_normalize_frame(sequence[frame_ind])
  return sequence

def check_normalization(sequence):
  global avg_dist
  if not config.APPLY_SEGMENT_DISTANCE_NORMALIZATION:
    raise ValueError
  for frame in sequence:
    assert np.array_equal(frame[3], np.array([0, 0, 0]))
    for (u, v) in skeleton_edges:
      assert avg_dist[u][v]!=0
      assert abs(get_distance(frame[u], frame[v]) - avg_dist[u][v])<=config.TOLERANCE, f"Distance did not match {get_distance(frame[u], frame[v])} and {avg_dist[u][v]}"