import random


class Frustum:
    def __init__(
        self,
        z_range,
        resolution,
        max_radius=0.6,
        dead_zone=0.05,
        focal_length=50,
        sensor_size=36,
    ):
        self.tan = (1 - dead_zone) * sensor_size / focal_length / 2
        self.offset = max_radius / self.tan * (self.tan ** 2 + 1) ** 0.5
        self.ratio = resolution[1] / resolution[0]
        self.z_range = z_range

    def gen_point(self, z=None):
        if z is None:
            z_min, z_max = self.z_range
            z = z_min + random.random() * (z_max - z_min)
        x = (random.random() * 2 - 1) * self.tan * (z + self.offset)
        y = (random.random() * 2 - 1) * self.tan * (z * self.ratio + self.offset)
        return x, y, z

    def gen_point_near(self, point, max_delta_z, delta_xy_range):
        x, y, z = point
        dxy_min, dxy_max = min(delta_xy_range) ** 2, max(delta_xy_range) ** 2
        for _ in range(100000):
            x2, y2, z2 = self.gen_point(z + max_delta_z * (random.random() * 2 - 1))
            if dxy_min <= (x - x2) ** 2 + (y - y2) ** 2 <= dxy_max:
                return x2, y2, z2
        raise ValueError("Failed to generate point in given range. Check parameters.")

    def gen_point_pair(self, max_delta_z, delta_xy_range):
        for _ in range(50):
            a = self.gen_point()
            try:
                b = self.gen_point_near(a, max_delta_z, delta_xy_range)
            except ValueError:
                continue
            return a, b
        raise ValueError("Failed to generate point pair. Check parameters.")
