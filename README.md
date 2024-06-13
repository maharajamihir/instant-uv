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
