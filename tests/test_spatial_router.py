"""
tests/test_spatial_router.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for app/geo/spatial_router.py

Tests cover:
  • Point-in-polygon correctness on mock GeoDataFrame.
  • India bounding-box guard (rejects out-of-country coordinates).
  • Boundary / edge-case coordinates.
  • Administrative hierarchy completeness.
"""

from __future__ import annotations

import pytest

from app.geo.spatial_router import (
    VillageSpatialIndex,
    _is_within_india,
    resolve_admin_hierarchy,
)


# ──────────────────────────────────────────────────────────────────────────────
# INDIA BOUNDING BOX GUARD
# ──────────────────────────────────────────────────────────────────────────────
class TestIndiaBoundingBox:
    @pytest.mark.parametrize("lat, lon, expected", [
        (25.5,  71.2,  True),   # Rajasthan — valid
        (17.7,  74.7,  True),   # Maharashtra — valid
        (26.2,  79.2,  True),   # UP — valid
        (51.5,  0.12,  False),  # London — outside India
        (-33.8, 151.2, False),  # Sydney — outside India
        (3.0,   101.0, False),  # KL — outside India
        (37.6,  69.0,  False),  # Just outside north bbox
        (6.0,   79.0,  False),  # Just outside south bbox
    ])
    def test_bounding_box(self, lat: float, lon: float, expected: bool) -> None:
        assert _is_within_india(lat, lon) == expected


# ──────────────────────────────────────────────────────────────────────────────
# POINT-IN-POLYGON ON MOCK DATA
# ──────────────────────────────────────────────────────────────────────────────
class TestPointInPolygon:
    """
    Uses the VillageSpatialIndex with force-loaded mock GeoDataFrame
    (the real GeoJSON is not present in tests — mock is loaded automatically).
    """

    @pytest.fixture(autouse=True)
    def index(self) -> VillageSpatialIndex:
        # Force mock data by pointing to a non-existent path
        self.idx = VillageSpatialIndex("nonexistent_path_triggers_mock.geojson")
        return self.idx

    def test_rajasthan_barmer_point(self) -> None:
        """GPS inside mock Sindhari polygon (Barmer, Rajasthan)."""
        result = self.idx.lookup(latitude=25.7, longitude=71.3)
        assert result is not None
        assert result["state_name"] == "Rajasthan"
        assert result["district_name"] == "Barmer"
        assert result["village_name"] == "Sindhari"

    def test_maharashtra_solapur_point(self) -> None:
        """GPS inside mock Mohol polygon (Solapur, Maharashtra)."""
        result = self.idx.lookup(latitude=17.7, longitude=74.7)
        assert result is not None
        assert result["state_name"] == "Maharashtra"
        assert result["district_name"] == "Solapur"

    def test_uttar_pradesh_bareilly_point(self) -> None:
        """GPS inside mock Fatehganj West polygon (Bareilly, UP)."""
        result = self.idx.lookup(latitude=26.2, longitude=79.2)
        assert result is not None
        assert result["state_name"] == "Uttar Pradesh"
        assert result["district_name"] == "Bareilly"

    def test_point_outside_all_polygons_returns_none(self) -> None:
        """GPS inside India's bounding box but outside all village polygons → None."""
        result = self.idx.lookup(latitude=20.0, longitude=78.0)
        assert result is None

    def test_hierarchy_completeness(self) -> None:
        """Every resolved hierarchy must have all 8 required keys."""
        required_keys = {
            "village_code", "village_name",
            "sub_district_code", "sub_district_name",
            "district_code", "district_name",
            "state_code", "state_name",
        }
        result = self.idx.lookup(latitude=25.7, longitude=71.3)
        assert result is not None
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in AdminHierarchy"
            assert result[key] != "", f"Key '{key}' is empty string"


# ──────────────────────────────────────────────────────────────────────────────
# resolve_admin_hierarchy PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
class TestResolveAdminHierarchy:
    def test_out_of_india_returns_none(self) -> None:
        result = resolve_admin_hierarchy(latitude=51.5, longitude=0.12)
        assert result is None

    def test_valid_india_coords_return_dict_or_none(self) -> None:
        """Should return a dict (if polygon matched) or None — never raise."""
        result = resolve_admin_hierarchy(latitude=25.7, longitude=71.3)
        assert result is None or isinstance(result, dict)

