# Awesome Interactive Techniques in 3D Scene Generation ![Awesome](https://awesome.re/badge.svg)


[![Paper](https://img.shields.io/badge/Paper-Authorea-b31b1b.svg)](https://doi.org/10.22541/au.177083718.88659470/v1)
[![DOI](https://img.shields.io/badge/DOI-10.22541%2Fau.177083718.88659470%2Fv1-blue)](https://doi.org/10.22541/au.177083718.88659470/v1)

</div>

If you find our survey or this repository helpful, please cite:

**A Comprehensive Survey of Interaction Techniques in 3D Scene Generation** *Yuqi Li, Siwei Meng, Chuanguang Yang, et al. (2026)* [[Read Paper]](https://doi.org/10.22541/au.177083718.88659470/v1)

---

A curated list of works on interactive techniques in 3D scene generation, including human-scene interaction, scene editing, layout-guided generation, and related datasets.

If you find missing works or have suggestions, welcome to open an issue or contact the maintainers.

---
## Content
---
- [Surveys](#surveys)
- [Datasets](#datasets)
- [Foundational Works](#foundational-works)
- [Human-Scene / Human-Object Interaction](#human-scene--human-object-interaction)
- [3D Scene Generation](#3d-scene-generation)
- [3D Scene Editing (NeRF / Gaussian Splatting)](#3d-scene-editing-nerf--gaussian-splatting)
- [4D / Dynamic Scene Generation](#4d--dynamic-scene-generation)
- [Text-to-3D / Image-to-3D](#text-to-3d--image-to-3d)
- [Others](#others)
---
## Surveys

1. \[arXiv 2024\] 3D representation methods: A survey [Paper](https://arxiv.org/abs/2410.06475)
2. \[arXiv 2025\] 3D human interaction generation: A survey [Paper](https://arxiv.org/pdf/2503.13120)
3. \[PAMI\] Human Motion Video Generation: A Survey [Paper](https://arxiv.org/pdf/2509.03883) [Code](https://github.com/Winn1y/Awesome-Human-Motion-Video-Generation)

---
## Datasets

1. \[3DV 2016\] SceneNN: A scene meshes dataset with annotations [Paper](https://ieeexplore.ieee.org/abstract/document/7785081/) [Code](http://www.scenenn.net)
2. \[CVPR 2016\] A benchmark dataset and evaluation methodology for video object segmentation [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.html) [Code](https://github.com/fperazzi/davis)
3. \[ICCV 2021\] 3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Fu_3D-FRONT_3D_Furnished_Rooms_With_layOuts_and_semaNTics_ICCV_2021_paper.pdf)
4. \[CVPR 2022\] BEHAVE: Dataset and method for tracking human object interactions [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Bhatnagar_BEHAVE_Dataset_and_Method_for_Tracking_Human_Object_Interactions_CVPR_2022_paper.html) [Code](http://virtualhumans.mpi-inf.mpg.de/behave)
5. \[CVPR 2023\] Objaverse: A universe of annotated 3D objects [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Deitke_Objaverse_A_Universe_of_Annotated_3D_Objects_CVPR_2023_paper.pdf) [Code](https://objaverse.allenai.org/docs/objaverse-xl)
6. \[CoRR 2024\] FORCE: Dataset and Method for Intuitive Physics Guided Human-object Interaction [Paper](https://openreview.net/forum?id=OUD5MPLauH) [Code](https://virtualhumans.mpi-inf.mpg.de/force/)
7. \[CVPR 2025\] CORE4D: A 4D human-object-human interaction dataset for collaborative object rearrangement [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CORE4D_A_4D_Human-Object-Human_Interaction_Dataset_for_Collaborative_Object_REarrangement_CVPR_2025_paper.html) [Code](https://core4d.github.io/)

---
## Foundational Works

1. \[NeurIPS 2014\] Generative adversarial nets [Paper](https://proceedings.neurips.cc/paper_files/paper/2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf)
2. \[ACM TOG 2018\] Language-driven synthesis of 3D scenes from scene databases [Paper](https://dl.acm.org/doi/abs/10.1145/3272127.3275035)
3. \[CVPR 2019\] DeepSDF: Learning continuous signed distance functions for shape representation [Paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html)
4. \[NeurIPS 2020\] Denoising diffusion probabilistic models [Paper](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) [Code](https://github.com/hojonathanho/diffusion)
5. \[Communications of the ACM 2021\] NeRF: Representing scenes as neural radiance fields for view synthesis [Paper](https://dl.acm.org/doi/abs/10.1145/3503250)
6. \[CVPR 2022\] High-Resolution Image Synthesis With Latent Diffusion Models [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) [Code](https://github.com/CompVis/latent-diffusion)
7. \[ACM Trans. Graph. 2023\] 3D Gaussian Splatting for Real-Time Radiance Field Rendering [Paper](https://sgvr.kaist.ac.kr/~sungeui/ICG_F23/Students/[CS482]%203D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%20Rendering.pdf)

---
## Human-Scene / Human-Object Interaction

1. \[ICCV 2019\] Resolving 3D Human Pose Ambiguities With 3D Scene Constraints [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hassan_Resolving_3D_Human_Pose_Ambiguities_With_3D_Scene_Constraints_ICCV_2019_paper.pdf) [Code](https://prox.is.tue.mpg.de)
2. \[ECCV 2020\] Long-term human motion prediction with scene contexts [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_23) [Code](http://virtualhumans.mpi-inf.mpg.de/behave)
3. \[ECCV 2020\] GRAB: A Dataset of Whole-Body Human Grasping of Objects [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_34) [Code](https://grab.is.tue.mpg.de/)
4. \[ICCV 2021\] Stochastic scene-aware motion prediction [Paper](http://openaccess.thecvf.com/content/ICCV2021/html/Hassan_Stochastic_Scene-Aware_Motion_Prediction_ICCV_2021_paper.html) [Code](https://samp.is.tue.mpg.de)
5. \[NeurIPS 2022\] HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6030db5195150ac86d942186f4abdad8-Abstract-Conference.html) [Code](https://silverster98.github.io/HUMANISE/)
6. \[arXiv 2023\] PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction [Paper](https://arxiv.org/abs/2312.04393) [Code](https://wyhuai.github.io/physhoi-page/)
7. \[arXiv 2023\] Unified Human-Scene Interaction via Prompted Chain-of-Contacts [Paper](https://arxiv.org/abs/2309.07918) [Code](https://github.com/OpenRobotLab/UniHSI)
8. \[ICCV 2023\] InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion [Paper](http://openaccess.thecvf.com/content/ICCV2023/html/Xu_InterDiff_Generating_3D_Human-Object_Interactions_with_Physics-Informed_Diffusion_ICCV_2023_paper.html) [Code](https://sirui-xu.github.io/InterDiff/)
9. \[CVPR 2024\] HOI-M3: Capture Multiple Humans and Objects Interaction within Contextual Environment [Paper](http://openaccess.thecvf.com/content/CVPR2024/html/Zhang_HOI-M3_Capture_Multiple_Humans_and_Objects_Interaction_within_Contextual_Environment_CVPR_2024_paper.html) [Code](https://juzezhang.github.io/HOIM3_ProjectPage/)
10. \[3DV 2024\] Synthesizing physically plausible human motions in 3D scenes [Paper](https://ieeexplore.ieee.org/abstract/document/10550906/) [Code](https://liangpan99.github.io/InterScene/)
11. \[SIGGRAPH Asia 2025\] Uni-Inter: Unifying 3D Human Motion Synthesis Across Diverse Interaction Contexts [Paper](https://dl.acm.org/doi/abs/10.1145/3757377.3763954)
12. \[SIGGRAPH Asia 2025\] PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image [Paper](https://dl.acm.org/doi/abs/10.1145/3757377.3763862) [Code](https://yuxuan-xue.com/physic)
13. \[CVPR 2025\] InteractMove: Text-Controlled Human-Object Interaction Generation in 3D Scenes with Movable Objects [Paper](https://dl.acm.org/doi/abs/10.1145/3746027.3754910) [Code](https://github.com/Cxhcmhhh/InteractMove)
14. \[arXiv 2024\] ZeroHSI: Zero-Shot 4D Human-Scene Interaction by Video Generation [Paper](https://arxiv.org/abs/2412.18600) [Code](https://awfuact.github.io/zerohsi)

---
## 3D Scene Generation

1. \[EMNLP 2021\] CLIPScore: A reference-free evaluation metric for image captioning [Paper](https://aclanthology.org/2021.emnlp-main.595/)
2. \[NeurIPS 2024\] SceneCraft: Layout-guided 3D scene generation [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/953d276d037e701fcd97dbb34ebb2394-Abstract-Conference.html) [Code](https://orangesodahub.github.io/SceneCraft)
3. \[arXiv 2024\] Toward Scene Graph and Layout Guided Complex 3D Scene Generation [Paper](https://arxiv.org/abs/2412.20473)
4. \[arXiv 2025\] Layout2Scene: 3D semantic layout guided scene generation via geometry and appearance diffusion priors [Paper](https://arxiv.org/pdf/2007.03672)
5. \[arXiv 2025\] SpatialGen: Layout-guided 3D indoor scene generation [Paper](https://arxiv.org/abs/2509.14981) [Code](https://manycore-research.github.io/SpatialGen)
6. \[arXiv 2025\] Controllable 3D outdoor scene generation via scene graphs [Paper](https://arxiv.org/abs/2503.07152)
7. \[CVPR 2025\] SceneFactor: Factored latent 3D diffusion for controllable 3D scene generation [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Bokhovkin_SceneFactor_Factored_Latent_3D_Diffusion_for_Controllable_3D_Scene_Generation_CVPR_2025_paper.html) [Code](https://github.com/alexeybokhovkin/SceneFactor)
8. \[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_WonderWorld_Interactive_3D_Scene_Generation_from_a_Single_Image_CVPR_2025_paper.html) [Code](https://kovenyu.com/WonderWorld/)
9. \[ICCV 2025\] Bolt3D: Generating 3D scenes in seconds [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Szymanowicz_Bolt3D_Generating_3D_Scenes_in_Seconds_ICCV_2025_paper.html) [Code](https://szymanowiczs.github.io/bolt3d)
10. \[T-PAMI\] DreamScene: 3D Gaussian-based End-to-end Text-to-3D Scene Generation [Paper](https://ieeexplore.ieee.org/abstract/document/11204013/) [Code](https://dreamscene-project.github.io)
11. \[ICASSP 2025\] Text-Guided Editable 3D City Scene Generation [Paper](https://ieeexplore.ieee.org/abstract/document/10889459)

---
## 3D Scene Editing (NeRF / Gaussian Splatting)

1. \[ICCV 2023\] Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions [Paper](http://openaccess.thecvf.com/content/ICCV2023/html/Haque_Instruct-NeRF2NeRF_Editing_3D_Scenes_with_Instructions_ICCV_2023_paper.html) [Code](https://instruct-nerf2nerf.github.io)
2. \[CoRR 2023\] 4D-Editor: Interactive object-level editing in dynamic neural radiance fields via 4D semantic segmentation [Paper](https://openreview.net/forum?id=EXojF2REgP) [Code](https://patrickddj.github.io/4D-Editor)
3. \[CVPR 2024\] Customize your NeRF: Adaptive Source Driven 3D Scene Editing via Local-Global Iterative Training [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/He_Customize_your_NeRF_Adaptive_Source_Driven_3D_Scene_Editing_via_CVPR_2024_paper.pdf) [Code](https://github.com/hrz2000/CustomNeRF)
4. \[ECCV 2024\] Chat-edit-3D: Interactive 3D scene editing via text prompts [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72946-1_12) [Code](https://sk-fun.fun/CE3D)
5. \[MM 2024\] iControl3D: An interactive system for controllable 3D scene generation [Paper](https://dl.acm.org/doi/abs/10.1145/3664647.3680557) [Code](https://github.com/xingyi-li/iControl3D)
6. \[arXiv 2024\] DragScene: Interactive 3D Scene Editing with Single-view Drag Instructions [Paper](https://arxiv.org/pdf/2412.13552)
7. \[TVCG\] GaussEdit: Adaptive 3D Scene Editing With Text and Image Prompts [Paper](https://ieeexplore.ieee.org/abstract/document/10947125/)
8. \[ICCV 2025\] InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wen_InterGSEdit_Interactive_3D_Gaussian_Splatting_Editing_with_3D_Geometry-Consistent_Attention_ICCV_2025_paper.html)

---
## 4D / Dynamic Scene Generation

1. \[CVPR 2024\] A Unified Approach for Text- and Image-guided 4D Scene Generation [Paper](http://openaccess.thecvf.com/content/CVPR2024/html/Zheng_A_Unified_Approach_for_Text-_and_Image-guided_4D_Scene_Generation_CVPR_2024_paper.html) [Code](https://research.nvidia.com/labs/nxp/dream-in-4d/)
2. \[ICML 2023\] Text-To-4D Dynamic Scene Generation [Paper](https://arxiv.org/abs/2301.11280) [Code](https://make-a-video3d.github.io/)
3. \[ECCV 2024\] Audio-Synchronized Visual Animation [Paper](https://arxiv.org/pdf/2403.05659) [Code](https://lzhangbj.github.io/projects/asva/asva.html)
4. \[SIGGRAPH Asia 2024\] Autonomous Character-Scene Interaction Synthesis from Text Instruction [Paper](https://dl.acm.org/doi/full/10.1145/3680528.3687595)
5. \[ACM ToG 2024\] BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation [Paper](https://dl.acm.org/doi/abs/10.1145/3658188)
6. \[MM 2024\] SceneExpander: Real-Time Scene Synthesis for Interactive Floor Plan Editing [Paper](https://dl.acm.org/doi/abs/10.1145/3664647.3680798) [Code](https://github.com/Shao-Kui/3DScenePlatform#sceneexpander)
7. \[arXiv 2025\] Drag4D: Align Your Motion with Text-Driven 3D Scene Generation [Paper](https://arxiv.org/abs/2509.21888)

---
## Text-to-3D / Image-to-3D

1. \[CVPR 2022\] Make It Move: Controllable Image-to-Video Generation With Text Descriptions [Paper](http://openaccess.thecvf.com/content/CVPR2022/html/Hu_Make_It_Move_Controllable_Image-to-Video_Generation_With_Text_Descriptions_CVPR_2022_paper.html) [Code](https://github.com/Youncy-Hu/MAGE)
2. \[CVPR 2023\] Magic3D: High-resolution text-to-3D content creation [Paper](http://openaccess.thecvf.com/content/CVPR2023/html/Lin_Magic3D_High-Resolution_Text-to-3D_Content_Creation_CVPR_2023_paper.html) [Code](https://research.nvidia.com/labs/dir/magic3d)
3. \[ICLR 2023\] DreamFusion: Text-to-3D using 2D diffusion [Paper](https://arxiv.org/abs/2209.14988) [Code](https://dreamfusion3d.github.io/)
4. \[ICCV 2025\] SegmentDreamer: Towards High-fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_SegmentDreamer_Towards_High-fidelity_Text-to-3D_Synthesis_with_Segmented_Consistency_Trajectory_Distillation_ICCV_2025_paper.html) [Code](https://zjhJOJO.github.io/segmentdreamer)
5. \[arXiv 2024\] Follow-your-click: Open-domain regional image animation via short prompts [Paper](https://arxiv.org/abs/2403.08268) [Code](https://follow-your-click.github.io/)
6. \[arXiv 2025\] TextMesh4D: High-quality text-to-4D mesh generation [Paper](https://arxiv.org/pdf/2506.24121)
7. \[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories [Paper](http://openaccess.thecvf.com/content/CVPR2025/papers/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.pdf) [Code](https://motion-prompting.github.io/)

---
## Others

1. \[arXiv 2024\] Task-oriented Sequential Grounding and Navigation in 3D Scenes [Paper](https://arxiv.org/abs/2408.04034) [Code](https://sg-3d.github.io/)
