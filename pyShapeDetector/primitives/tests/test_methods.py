import os
import numpy as np
from numpy.testing import assert_almost_equal
from pyShapeDetector.primitives import PlaneBounded


def test_get_bounded_planes_from_grid():
    path = os.path.join(
        os.path.dirname(__file__), "data", "plane_test_get_bounded_planes_from_grid.tar"
    )
    plane_input = PlaneBounded.load(path)

    num_grid_points = 1000
    k = 15
    eps = plane_input.inliers.average_nearest_dist(k=k)

    if num_grid_points is not None:
        grid_width = np.sqrt(plane_input.surface_area / num_grid_points)
    else:
        grid_width = eps / 2

    concave_planes = plane_input.get_bounded_planes_from_grid(
        grid_width=grid_width,
        max_point_dist=eps * 2,
        grid_type="regular",
        perimeter_multiplier=1.5,
        return_rect_grid=False,
        perimeter_eps=1e-3,
        only_inside=False,
        add_boundary=False,
        detect_holes=False,
        add_inliers=True,
        angle_colinear=np.deg2rad(0),
        colinear_recursive=True,
        contract_boundary=True,
        min_inliers=1,
        max_grid_points=20000,
    )

    assert len(concave_planes) == 2
    for plane in concave_planes:
        assert_almost_equal(plane.surface_area, plane.mesh.get_surface_area())
    assert_almost_equal(concave_planes[0].surface_area, 1.5326484340740056)
    assert_almost_equal(concave_planes[1].surface_area, 0.010965917111882995)
