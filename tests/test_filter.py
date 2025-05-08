import pytest
import numpy as np
import pandas as pd

# Import all the filter functions
from CPET.utils.io import (
    filter_radius,
    filter_radius_whole_residue,
    filter_IDs,
    filter_residue,
    filter_resnum,
    filter_resnum_andname,
    filter_in_box,
    filter_atom_num,
)


class TestFilter:
    @pytest.fixture
    def synthetic_data(self):
        """
        Create synthetic data that roughly mimics the inputs
        your filter functions expect. We'll return a dictionary
        of arrays for convenience.
        """
        # Coordinates
        x = np.array(
            [
                [0.0, 0.0, 0.0],  # A
                [1.5, 0.0, 0.0],  # B
                [2.1, 1.0, 0.0],  # C
                [5.0, 5.0, 5.0],  # D
            ],
            dtype=float,
        )

        # Charge array
        Q = np.array([+1.0, -0.5, +0.3, +1.2], dtype=float)

        # Residue names and numbers
        resids = np.array(["WAT", "WAT", "FE", "NI"])  # Example residue IDs
        resnums = np.array([10, 10, 651, 652])  # Example residue numbers

        # Atom info
        atom_numbers = np.array([1001, 1002, 2001, 3001])  # Unique identifiers
        atom_types = np.array(["O", "H", "FE", "NI"])

        # Combine them into a single dictionary to pass around
        data = {
            "x": x,
            "Q": Q,
            "resids": resids,
            "resnums": resnums,
            "atom_number": atom_numbers,
            "atom_type": atom_types,
        }
        return data

    def test_filter_radius(self, synthetic_data):
        """Test filter_radius with a simple radius around origin."""
        x = synthetic_data["x"]
        Q = synthetic_data["Q"]

        center = np.array([0.0, 0.0, 0.0], dtype=float)
        radius = 2.0

        x_filtered, Q_filtered = filter_radius(x, Q, center, radius=radius)
        # We expect points within radius=2.0 to remain.
        # Let's see which of our synthetic data are within distance 2.0 of [0,0,0].
        #  -- [0.0, 0.0, 0.0] is distance 0.0  -> inside
        #  -- [1.5, 0.0, 0.0] is distance 1.5  -> inside
        #  -- [2.1, 1.0, 0.0] is distance sqrt(2.1^2 + 1^2) ~ 2.33 -> outside
        #  -- [5.0, 5.0, 5.0] is distance sqrt(75) ~ 8.66 -> outside
        # So we expect 2 points to remain.
        assert len(x_filtered) == 2
        # The first two charges should remain
        np.testing.assert_allclose(x_filtered[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(x_filtered[1], [1.5, 0.0, 0.0])
        # Corresponding Q values
        np.testing.assert_allclose(Q_filtered, [1.0, -0.5])

    def test_filter_radius_whole_residue(self, synthetic_data):
        """
        Test filter_radius_whole_residue: if *any* atom of a residue is
        outside the radius, the entire residue is removed.
        """
        x = synthetic_data["x"]
        Q = synthetic_data["Q"]
        resids = synthetic_data["resids"]
        resnums = synthetic_data["resnums"]

        center = np.array([0.0, 0.0, 0.0], dtype=float)
        radius = 2.0
        # We'll pretend each position is from a single residue.
        # But note that in your real data, multiple atoms can share
        # the same residue number & name. We'll just use the data as is.

        x_filt, Q_filt = filter_radius_whole_residue(
            x, Q, resids, resnums, center, radius=radius
        )
        # Explanation:
        #  - The first 2 entries are residue WAT #10; both are within distance < 2.0
        #    => that residue might remain if *all* points for WAT #10 are within radius.
        #    In this example, we do indeed have both points within ~1.5, so they're inside.
        #  - The 3rd entry is FE #651, at distance ~2.3 => outside => entire residue #651
        #    gets removed
        #  - The 4th entry is NI #652, definitely outside => entire residue #652 gets removed
        #
        # So we only expect to keep the points for residue #10 (the first 2).
        assert len(x_filt) == 2
        np.testing.assert_allclose(x_filt[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(x_filt[1], [1.5, 0.0, 0.0])
        np.testing.assert_allclose(Q_filt, [1.0, -0.5])

    def test_filter_IDs(self, synthetic_data):
        """
        Test filter_IDs, which uses a DataFrame and a dictionary of
        potential ID-based filters.
        """
        from collections import namedtuple

        # Build the ID list as (atom_number, atom_type, resid, resnum, chain)
        # We'll just use chain='A' for simplicity.
        # You could also keep chain empty if not used.
        ID = []
        for i in range(len(synthetic_data["x"])):
            ID.append(
                (
                    synthetic_data["atom_number"][i],
                    synthetic_data["atom_type"][i],
                    synthetic_data["resids"][i],
                    synthetic_data["resnums"][i],
                    "A",  # chain
                )
            )

        # Let's create a filter_dict that tries to remove any FE or NI atoms
        # We need to provide them in lists of equal length. We'll do just 1 filter row.
        filter_dict = {
            "atom_type": ["FE"],  # remove all atom_type == "FE"
            "resid": [""],  # not filtering by resid
            "resnum": [""],  # not filtering by resnum
            "chain": [""],  # not filtering by chain
        }

        x_filtered, Q_filtered, ID_filtered = filter_IDs(
            synthetic_data["x"], synthetic_data["Q"], ID, filter_dict
        )

        # In synthetic_data, we have 4 atoms: O (WAT#10), H (WAT#10), FE(#651), NI(#652)
        # "atom_type":"FE" means we remove the FE entry. But we did NOT specify "NI",
        # so NI stays.
        # Original length was 4, so let's see what's removed:
        #   - FE is removed
        #   - O, H, and NI remain
        # => we expect 3 atoms left
        assert len(x_filtered) == 3
        # Let's verify that the FE item was removed:
        # The FE item was the 3rd index in the original arrays:
        #   x[2] = [2.1, 1.0, 0.0]
        # We check that what's left doesn't contain that coordinate:
        for coord in x_filtered:
            assert not np.allclose(coord, [2.1, 1.0, 0.0])

        # We can also check that none of the remaining ID entries have "FE" in them
        for row in ID_filtered:
            # row = (atom_number, atom_type, resid, resnum, chain)
            assert row[1] != "FE"

    def test_filter_residue(self, synthetic_data):
        """
        Test filter_residue by removing e.g. 'WAT' from the dataset.
        """
        # We'll call the function: filter_residue(x, Q, resnums, resids, atom_number, atom_type, filter_list)
        x, Q, resnums, resids, anums, atypes = [
            synthetic_data[k]
            for k in ["x", "Q", "resnums", "resids", "atom_number", "atom_type"]
        ]

        filter_list = ["WAT"]  # remove all "WAT" residues
        (x_filt, Q_filt, resnums_filt, resids_filt, anums_filt, atypes_filt) = (
            filter_residue(x, Q, resnums, resids, anums, atypes, filter_list)
        )

        # Original data had WAT #10 for the first two items => removed.
        # FE #651, NI #652 remain => we expect 2 items left
        assert len(x_filt) == 2
        # They should match the last two from the original
        np.testing.assert_allclose(x_filt[0], [2.1, 1.0, 0.0])
        np.testing.assert_allclose(x_filt[1], [5.0, 5.0, 5.0])

    def test_filter_resnum(self, synthetic_data):
        """
        Test filter_resnum by removing e.g. residue #651.
        """
        x, Q, resnums, resids = [
            synthetic_data[k] for k in ["x", "Q", "resnums", "resids"]
        ]
        filter_list = [651]  # remove residue number 651
        x_filt, Q_filt, resnums_filt, resids_filt = filter_resnum(
            x, Q, resnums, resids, filter_list
        )

        # Residue #651 was the 3rd entry (FE).
        # So we remove that => expect 3 left
        assert len(x_filt) == 3
        # Let's check the coordinates are the 1st, 2nd, and 4th from the original
        # which are [0,0,0], [1.5,0,0], and [5,5,5].
        np.testing.assert_allclose(x_filt[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(x_filt[1], [1.5, 0.0, 0.0])
        np.testing.assert_allclose(x_filt[2], [5.0, 5.0, 5.0])

    def test_filter_resnum_andname(self, synthetic_data):
        """
        Test filter_resnum_andname, which removes atoms with a specific (resnum -> resname) mapping.
        """
        x = synthetic_data["x"]
        Q = synthetic_data["Q"]
        resnums = synthetic_data["resnums"]
        resnames = synthetic_data["resids"]  # 'resids' are effectively residue names
        anums = synthetic_data["atom_number"]
        atypes = synthetic_data["atom_type"]

        # Suppose we want to remove the pair { "651":"FE" } from the data
        filter_list = [{"651": "FE"}]
        (x_filt, Q_filt, resnums_filt, resnames_filt, anums_filt, atypes_filt) = (
            filter_resnum_andname(x, Q, resnums, resnames, anums, atypes, filter_list)
        )

        # This should remove whichever row has resnum=651 with resname="FE", i.e. the 3rd row
        assert len(x_filt) == 3
        # Check the 3rd coordinate is gone
        for coord in x_filt:
            assert not np.allclose(coord, [2.1, 1.0, 0.0])

    def test_filter_in_box(self, synthetic_data):
        """
        Test filter_in_box, which *removes* points *inside* the given box.
        """
        x = synthetic_data["x"]
        Q = synthetic_data["Q"]

        # Suppose center is [1.0, 0.0, 0.0] and box is [1.5, 1.5, 1.5]
        center = np.array([1.0, 0.0, 0.0])
        dimensions = [1.5, 1.5, 1.5]

        x_filt, Q_filt = filter_in_box(x, Q, center, dimensions)
        # Points "inside" that box around center [1,0,0] in x±1.5, y±1.5, z±1.5 get removed.
        #
        # Let's see which ones are inside:
        #  Shift by center => x' = x - [1,0,0]:
        #    [0,0,0] -> [-1,0,0], still within x±1.5 => yes, y±1.5 => yes, z±1.5 => yes
        #    [1.5,0,0] -> [0.5,0,0], still within ±1.5 => yes
        #    [2.1,1.0,0] -> [1.1,1.0,0], x=1.1 <1.5 => inside
        #    [5,5,5] -> [4,5,5], x=4 >1.5 => outside
        #
        # So the first 3 points are "inside", and the last is "outside".
        # The function *removes* the inside points and keeps the outside ones => keep #4 only.
        assert len(x_filt) == 1
        np.testing.assert_allclose(x_filt[0], [5.0, 5.0, 5.0])
        np.testing.assert_allclose(Q_filt[0], 1.2)

    def test_filter_atom_num(self, synthetic_data):
        """
        Test filter_atom_num, which removes coordinates if their atom_number is in a filter_list.
        """
        x = synthetic_data["x"]
        Q = synthetic_data["Q"]
        anums = synthetic_data["atom_number"]

        # Suppose we want to remove the 2nd and 3rd entries => that corresponds to
        # atom_number=1002, 2001 in our synthetic data.
        filter_list = [1002, 2001]
        x_filt, Q_filt = filter_atom_num(x, Q, anums, filter_list)

        # The original had 4. We remove 2 => expect 2 remain.
        assert len(x_filt) == 2
        # We should keep only entries with atom_number=1001 and 3001 => indices 0,3 from original
        np.testing.assert_allclose(x_filt[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(x_filt[1], [5.0, 5.0, 5.0])
