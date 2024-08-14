import numpy as np
import trimesh
import xatlas
import matplotlib.pyplot as plt
import cv2

"""
Run this command before using this script:

curl https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj > stanford_bunny.obj
"""

# Load the Stanford Bunny mesh
mesh = trimesh.load('stanford_bunny.obj')

# Convert the mesh to the format expected by xatlas
vertices = np.array(mesh.vertices, dtype=np.float32)
faces = np.array(mesh.faces, dtype=np.uint32)

# Create a new xatlas atlas
atlas = xatlas.Atlas()

# Add the mesh to the atlas
atlas.add_mesh(vertices, faces)

# Generate the UVs
atlas.generate()

# Get the resulting mesh with UVs
result = atlas[0]

# Create a UV map image
uvs = result[2]
indices = result[1]
#breakpoint()
# Normalize UV coordinates to [0, 1]
uvs[:, 0] = (uvs[:, 0] - np.min(uvs[:, 0])) / (np.max(uvs[:, 0]) - np.min(uvs[:, 0]))
uvs[:, 1] = (uvs[:, 1] - np.min(uvs[:, 1])) / (np.max(uvs[:, 1]) - np.min(uvs[:, 1]))

# Create an image to visualize the UV map
img_size = 1024*4
uv_image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

for face in indices:
    pts = uvs[face] * img_size
    pts = pts.astype(np.int32)
    pts = pts.reshape((-1,1,2))
    uv_image = cv2.polylines(uv_image, [pts], isClosed=True, color=(0, 0, 0), thickness=5)

# Save the UV map as an image
plt.imshow(uv_image)
plt.axis('off')
plt.savefig('uv_map_bunny.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
