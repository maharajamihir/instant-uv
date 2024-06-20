# InstantUV
InstantUV: Fast Implicit Texture Learning for 2D Mesh Rendering

Instant-NGP [2] was a breakthrough paper, significantly speeding up the learning and inference of
implicit representations. Yet, as said in the lecture, despite the growth in implicit representations,
meshes remain a preferred choice in many downstream applications. This is due to their intuitive
editing capabilities and the existence of efficient algorithms. Representing neural fields (like texture
(color)) on meshes still remains an open task. Storing texture on a mesh boils down to either storing
it on the vertices and interpolating for the faces, storing one color per face, or storing it in an image
using a UV map. We propose a novel method to store texture implicitly by unwrapping the mesh
into an image using UV mapping and learning the stored texture in the image implicitly in 2D using
Instant-NGP [2] (their gigapixel image part). Related work that learns texture on a mesh exists
[3, 4, 1], but ours would be way more efficient, and it would be interesting to see if reducing the
problem to a 2D learning task solves it more effectively.

# Work with InstantUV

## Install packages

```bash
# I have tested it with python=3.9, but might also work with newer version
conda create -n instantuv python=3.9
conda activate instantuv
```

```bash
pip install -r requirements.txt
pip install pyembree # not sure why this fails when downloading through requirements.txt
```

## Download the data
The data will be downloaded under `data/raw`. Please don't change its download location, as the preprocessing script will pick it up from there. 
```bash
./src/data/download_data.sh
```

## Preprocess the data
Preprocess the data by performing ray-mesh intersection given the images and the mesh object. Here we separately preprocess train, val and test data, for which we defined the split beforehand. The following command preprocesses the data for the human object for the train split. Do this for the `val` split as well and repeat the process for the `cat` object. Note, that we dont need to preprocess the test dataset, since we run our evaluations on rendered images directly.
```bash
python src/data/preprocess_dataset.py --config_path config/human/config_human.yaml --split train
```
Tipp: Run the preprocessing with the [`nice`](https://man7.org/linux/man-pages/man2/nice.2.html) command, since it is very cpu intensive and your laptop might crash. 


## Train model


## Visualize

## Run experiments


# Notes
## Pipeline 

We have the untextured 3d object as a mesh and we multiview images of our object (colored).

(mesh, images) -> (3d coordinates, rgb value) -> 

-> 3d coordinate -> unwrap mesh into UV coordinates -> (3d coords, 2d coordinates, rgb values)
 
 f(3d coordinate) = rgb value

 h(3d coordinate) = 2d coodinate
 g(2d coodinate) = rgb value

 g(h(x)) = f(x)

<hr/>

### References

[1] Lukas Koestler, Daniel Grittner, Michael Moeller, Daniel Cremers, and Zorah Lähner. Intrinsic
neural fields: Learning functions on manifolds. In European Conference on Computer Vision,
pages 622–639. Springer, 2022. 

[2] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics
primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):
1–15, 2022.

[3] Michael Oechsle, Lars Mescheder, Michael Niemeyer, Thilo Strauss, and Andreas Geiger. Texture
fields: Learning texture representations in function space. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 4531–4540, 2019.

[4] Fanbo Xiang, Zexiang Xu, Milos Hasan, Yannick Hold-Geoffroy, Kalyan Sunkavalli, and Hao
Su. Neutex: Neural texture mapping for volumetric neural rendering. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7119–7128, 2021.
