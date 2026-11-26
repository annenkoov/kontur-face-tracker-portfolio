import unittest

from face_tracker.ui import responsive_wraplength


class ResponsiveWrapLengthTests(unittest.TestCase):
    def test_wrap_length_never_exceeds_original_design_cap(self):
        self.assertEqual(responsive_wraplength(340, 900), 340)

    def test_wrap_length_shrinks_to_the_available_grid_cell(self):
        self.assertEqual(responsive_wraplength(340, 195), 195)

    def test_wrap_length_keeps_a_small_usable_floor(self):
        self.assertEqual(responsive_wraplength(340, 30), 80)


if __name__ == "__main__":
    unittest.main()
