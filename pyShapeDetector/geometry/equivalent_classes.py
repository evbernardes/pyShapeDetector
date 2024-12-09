from open3d import geometry

from .pointcloud import PointCloud
from .trianglemesh import TriangleMesh
from .axis_aligned_bounding_box import AxisAlignedBoundingBox
from .oriented_bounding_box import OrientedBoundingBox
from .lineset import LineSet

equivalent_classes_dict = {
    geometry.PointCloud: PointCloud,
    geometry.TriangleMesh: TriangleMesh,
    geometry.AxisAlignedBoundingBox: AxisAlignedBoundingBox,
    geometry.OrientedBoundingBox: OrientedBoundingBox,
    geometry.LineSet: LineSet,
}
