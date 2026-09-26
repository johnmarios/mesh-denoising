# normalize bunny_low.obj and save it at resources/q6/bunny_low_normalized.obj
import os
import numpy as np
from mesh_denoising_q7_q8_clean.core.mesh import prepare_mesh, create_mesh, copy_mesh, vertex_normals, save_mesh

def main():

    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(root_dir, "resources", "bunny_low.obj")
    output_path = os.path.join(root_dir, "resources", "q6", "bunny_low_normalized.obj")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    mesh = prepare_mesh(create_mesh(input_path))
    save_mesh(mesh, output_path)

if __name__ == "__main__":
    main()