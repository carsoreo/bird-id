import unittest

from lightcraft_autoedit import denoise_amount, old_subject_kws, plan_crop


def frame(birds=(), subject=None, w=6000, h=4000):
    e = {"width": w, "height": h, "birds": [{"conf": c, "box": b} for c, b in birds]}
    if subject:
        e["subject"] = {"conf": 1.0, "box": subject}
    return e


class PlanCrop(unittest.TestCase):
    def test_no_bird_no_subject_leaves_frame_alone(self):
        self.assertEqual(plan_crop(frame()), ("none", None, 1.0))

    def test_big_bird_fills_frame(self):
        style, rect, mag = plan_crop(frame([(0.9, [0.4, 0.4, 0.6, 0.6])]))
        self.assertEqual(style, "fill")
        self.assertGreater(mag, 1)
        x0, y0, x1, y1 = rect
        self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1)
        self.assertAlmostEqual((x1 - x0) * 6000 / ((y1 - y0) * 4000), 3 / 2, places=2)

    def test_small_bird_keeps_habitat(self):
        style, rect, _ = plan_crop(frame([(0.9, [0.5, 0.5, 0.55, 0.55])]))
        self.assertEqual(style, "scene")
        self.assertGreaterEqual(rect[2] - rect[0], 0.30)  # never tighter than MIN_CROP

    def test_crop_stays_inside_frame_at_the_edge(self):
        _, rect, _ = plan_crop(frame([(0.9, [0.9, 0.9, 1.0, 1.0])]))
        self.assertLessEqual(rect[2], 1)
        self.assertLessEqual(rect[3], 1)

    def test_portrait_frame_gets_portrait_crop(self):
        _, rect, _ = plan_crop(frame([(0.9, [0.4, 0.4, 0.6, 0.55])], w=4000, h=6000))
        self.assertAlmostEqual((rect[2] - rect[0]) * 4000 / ((rect[3] - rect[1]) * 6000), 2 / 3, places=2)

    def test_weak_second_bird_is_ignored(self):
        one = plan_crop(frame([(0.9, [0.4, 0.4, 0.6, 0.6])]))
        two = plan_crop(frame([(0.9, [0.4, 0.4, 0.6, 0.6]), (0.2, [0.0, 0.0, 0.1, 0.1])]))
        self.assertEqual(one, two)

    def test_other_subject_gets_gentle_recompose(self):
        style, _, _ = plan_crop(frame(subject=[0.4, 0.4, 0.5, 0.5]))
        self.assertEqual(style, "subject")


class Denoise(unittest.TestCase):
    def test_steps_with_iso(self):
        self.assertEqual([denoise_amount(i, 1) for i in (400, 1000, 2000, 4000, 12800)], [0, 20, 35, 50, 65])

    def test_heavy_crop_adds_noise_reduction_and_caps(self):
        self.assertEqual(denoise_amount(400, 3), 25)
        self.assertEqual(denoise_amount(25600, 3), 80)


class SubjectKeywords(unittest.TestCase):
    def test_reads_list_or_comma_string(self):
        self.assertEqual(old_subject_kws({"meta": {"keywords": "Bird, Subject: Deer"}}), [" Subject: Deer"])
        self.assertEqual(old_subject_kws({"meta": {"keywords": ["Subject: Fox", "x"]}}), ["Subject: Fox"])
        self.assertEqual(old_subject_kws({"meta": {}}), [])


if __name__ == "__main__":
    unittest.main()
