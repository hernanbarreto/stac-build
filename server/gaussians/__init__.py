"""The Gaussian-Splatting realism layer over the metric product (USER 2026-10-31: "el realismo
de los GS con la capacidad de medir sobre malla matemáticamente buena como la nuestra … en la
etapa final … entrenar los GS y superponerlos … sobre el modelo GS debe poder seleccionarse los
objetos tal como lo hacemos actualmente").

Two layers in ONE world frame: the certified cloud / mesh MEASURES, the splats are what is seen.
The splats are trained at the end of the chain on what the chain certified and never move it:

  * the poses and the camera are the certified ones (``camera_poses.txt`` / ``camera.json`` of the
    live epoch) and stay FIXED — gsplat never optimises them;
  * the gaussians are INITIALISED from the published cloud (``cleaned_cloud.ply``: position, colour,
    and the per-point instance id of the projection, ``instance_ids.npy``), a stable subsample;
  * the depth the chain measured (the per-keyframe maps the TSDF chain reads, correction sidecar
    applied) supervises the rendered depth, so the splats sit on the measured surface;
  * a per-keyframe affine exposure is learned (the phone's auto exposure), held-out keyframes give
    a MEASURED PSNR / SSIM in the report — never a verdict.

Packages: ``config`` (the strict ``gaussians:`` section), ``plyio`` (the cloud reader, the 3DGS PLY
and the .splat writer), ``dataset`` (the training set written by the da3 worker), ``label`` (the
instance of every gaussian), ``train`` (gsplat, env ``gs`` — Apache-2.0; the Inria 3DGS code is
non-commercial and is never used), ``api`` (the viewer's routes). Worker: ``workers.gaussians_worker``.
"""
