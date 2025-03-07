### leaphand.urdf
This is a customized URDF modified from leaphand.urdf (generated from leaphand.urdf.xacro)
We made the following changes:
- re-name the link names from `leap_link_0` to `leap_link_15` (mcp/pip/dip/fingertip) and `leap_link_16` to `leap_link_19` (fingertip_new)
- change the asset root to `package://drake_models/leap_description/meshes`, thus can be used by Drake
