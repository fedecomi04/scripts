"""Vendored NDP (Neural Deformation Pyramid) no-learned model.

Minimal copy of github.com/rabbityl/DeformationPyramid model code (nets.py +
rigid_body.py only) so the hierarchical Sim3 deformation optimizer runs
in-process in the main dynamic_gs env (torch + numpy, no pytorch3d/open3d).
See dynamic_gs/utils/ndp_register.py for the registration wrapper.
"""
