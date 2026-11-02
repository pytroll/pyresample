"""Test the GCPDefinition class."""
import numpy as np
import pytest

from pyresample.geometry import GCPDefinition

gcp_dtype = np.dtype([("longitude", float),
                      ("latitude", float),
                      ("altitude", float),
                      ("x", float),
                      ("y", float)])


def test_coarse_bounding_box(gcp_definition):
    """Test the coarse bounding box method."""
    bblons = np.hstack((np.arange(10), np.arange(19, 99, 10), np.arange(99, 90, -1), np.arange(90, 0, -10)))
    np.testing.assert_array_equal(bblons, gcp_definition.get_coarse_bbox_lonlats()["longitude"])


@pytest.fixture
def gcp_definition():
    """Create a GCPDefinition instance."""
    lons = None
    lats = None
    gcps = np.zeros(100, dtype=gcp_dtype)
    gcps["longitude"] = np.arange(100)
    gcps["latitude"] = np.arange(100, 200)
    gcps = gcps.reshape((10, 10))
    gdef = GCPDefinition(lons, lats, gcps)
    return gdef
