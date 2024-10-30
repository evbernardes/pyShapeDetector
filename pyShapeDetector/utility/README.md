# Utility

 This submodule helper utilities.

 ## Available classes
 - `MultiDetector`: Assembles multiple detectors to run sequentially.
 - `PrimitiveLimits`: Utility to further filter possible fits in different `Primitive` types.
 - `ElementSelector`: Open3D-based interactive visualizer that allows for the manual selection of elements.
 - `DetectorOptions`: Base class used to define options for detector methods.

## Helper functions
This module also include multiple helper functions.

### Math helpers
- midrange
- get_area_with_shoelace
- check_vertices_clockwise
- get_rotation_from_axis
- rgb_to_cielab
- cielab_to_rgb

### Visualization helpers
- get_painted
- get_open3d_geometries
- draw_geometries
- draw_two_columns
- select_manually
- apply_function_manually
- select_combinations_manually (**SHOULDN'T BE USED ANYMORE**)

### Input/Output helpers
- mesh_to_obj_description
- write_obj
- create_unity_package
- check_existance
- save_elements
- save_ask
- ask_and_save
- 
### Internal helpers
- _set_and_check_3d_array
- combine_indices_to_remove
- parallelize
- accept_one_or_multiple_elements