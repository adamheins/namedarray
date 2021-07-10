import numpy as np

from namedarray import namedarray


def norm(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


class Vec3NameOnly:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        return norm(self.x, self.y, self.z)


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

    def test_steal_func(self):
        # we can also pretty easily translate general functions to use with our
        # custom classes
        Vec3.norm = lambda self: norm(self.x, self.y, self.z)
        v = Vec3([1, 0, 0])
        assert np.isclose(v.norm(), 1)

    def test_apply_numpy_func(self):
        v = Vec3([0, 1, 2])

        # numpy functions can be directly applied to NamedArrays
        assert np.allclose(np.sin(v), np.sin(v.array))

    def test_fromobj_all_fields(self):
        v1 = Vec3NameOnly(x=1, y=2, z=3)

        # create a Vec3, taking all fields from v1
        v2 = Vec3.fromobj(v1)

        assert v2.x == v1.x
        assert v2.y == v1.y
        assert v2.z == v1.z

    def test_fromobj_one_field(self):
        v1 = Vec3NameOnly(x=1, y=2, z=3)

        # now we only take the x field; y and z should be their default values
        # (zero)
        v2 = Vec3.fromobj(v1, fields=["x"])

        assert v2.x == v1.x
        assert v2.y == 0
        assert v2.z == 0


class TestSliceRange:
    def test_default(self):
        twist = Twist()
        assert type(twist.linear) == Vec3
        assert type(twist.angular) == Vec3
        assert np.allclose(twist.array, np.zeros(6))

    def test_cached_attr(self):
        # the object returned by each attribute should be the same every call
        # (we don't want to create a new one every time)
        twist = Twist()
        v1 = twist.linear
        v2 = twist.linear
        assert id(v1) == id(v2)
