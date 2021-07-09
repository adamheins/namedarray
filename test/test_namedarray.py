import numpy as np

from namedarray import namedarray


class Vec3NameOnly:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


Vec3 = namedarray(
    "Vec3",
    (3,),
    {
        "x": 0,
        "y": 1,
        "z": 2,
    },
)

Twist = namedarray(
    "Twist",
    (6,),
    {
        "linear": Vec3.slice(np.s_[:3]),
        "angular": Vec3.slice(np.s_[3:]),
    },
)


class TestIndexOnly:
    def test_default(self):
        v = Vec3()
        assert np.allclose(v.array, np.zeros(3))

    def test_get_attr(self):
        v = Vec3([0, 1, 2])
        assert v.x == 0
        assert v.y == 1
        assert v.z == 2

    def test_set_attr(self):
        v = Vec3()
        assert v.x == 0

        v.x = 5
        assert v.x == 5

    def test_steal_method(self):
        # steal the norm method from another class that uses the same
        # attributes
        Vec3.norm = Vec3NameOnly.norm
        v = Vec3([1, 0, 0])
        assert np.isclose(v.norm(), 1)

    def test_apply_numpy_func(self):
        v = Vec3([0, 1, 2])
        assert np.allclose(np.sin(v), np.sin(v.array))


class TestSliceRange:
    def test_default(self):
        twist = Twist()
        assert type(twist.linear) == Vec3
        assert type(twist.angular) == Vec3
        assert np.allclose(twist.array, np.zeros(6))

    def test_cached_attr(self):
        twist = Twist()
        v1 = twist.linear
        v2 = twist.linear
        assert id(v1) == id(v2)
