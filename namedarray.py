from sys import getsizeof
from functools import lru_cache, partial
from dataclasses import dataclass

import numpy as np
import IPython


@dataclass
class namedslice:
    name: str
    index: slice
    initializer: object = None


def namedarray(
    typename,
    shape,
    named_slices,
    methods=None,
):
    """Returns a new class with names for array slices.

    The new class wraps a numpy ndarray and allows getting and setting
    arbitrary slices by name."""

    def __init__(self, array=None):
        if array is None:
            self.array = np.zeros(shape)
        else:
            self.array = np.array(array, copy=False)
        assert self.array.shape == shape

    def __array__(self):
        return self.array

    def __repr__(self):
        return f"{typename}({list(self.array)})"

    def fillobj(self, obj, fields=None):
        """Fill field in objects from this array."""
        if fields is None:
            fields = cls._names

        for field in fields:
            if hasattr(obj, field):
                myvalue = getattr(self, field)
                try:
                    myvalue.fillobj(obj, field)
                except AttributeError:
                    setattr(obj, field, myvalue)
            else:
                raise AttributeError()

    @classmethod
    def fromobj(cls, obj, fields=None):
        """Generate array using the values of the fields in `obj`."""
        if fields is None:
            fields = cls._names

        self = cls()
        for field in fields:
            objvalue = getattr(obj, field)
            try:
                myvalue = getattr(self, field)
                setattr(self, field, myvalue.fromobj(objvalue))
            except AttributeError as e:
                setattr(self, field, objvalue)
        return self

    @classmethod
    def slice(cls, name, index):
        return namedslice(name=name, index=index, initializer=cls)

    cls = type(
        typename,
        (),
        {
            "__init__": __init__,
            "__array__": __array__,
            "__repr__": __repr__,
            "fillobj": fillobj,
            "fromobj": fromobj,
            "slice": slice
        },
    )
    cls._names = []

    def fget_initialized(self, index, initializer):
        return initializer(self.array[index])

    def fget(self, index):
        return self.array[index]

    def fset(self, value, index):
        self.array[index] = np.array(value, copy=False)

    # Create the properties on the objects, each of which gets/sets a
    # particular slice of the underlying array
    for named_slice in named_slices:
        name = named_slice.name
        cls._names.append(name)

        index = named_slice.index
        initializer = named_slice.initializer

        setter = partial(fset, index=index)

        if initializer is None:
            getter = partial(fget, index=index)
        else:
            # if the property gets initialized to another object, we don't want
            # to recreate it each time, so we cache it
            getter = lru_cache(maxsize=None)(
                partial(fget_initialized, index=index, initializer=initializer)
            )

        setattr(cls, name, property(fget=getter, fset=setter))

    return cls


class TestVec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class TestTwist:
    def __init__(self, linear, angular):
        self.linear = linear
        self.angular = angular


# def cached_property(func):
#     return property(lru_cache(maxsize=None)(func))


def main():
    test_twist = TestTwist(TestVec3(x=1), TestVec3(x=2))
    test_twist2 = TestTwist(np.array([0, 1, 2]), np.array([3, 4, 5]))

    Vec3 = namedarray(
        "Vec3",
        (3,),
        [
            namedslice(name="x", index=0),
            namedslice(name="y", index=1),
            namedslice(name="z", index=2),
        ],
    )
    v = Vec3([0, 1, 2])

    Twist = namedarray(
        "Twist",
        (6,),
        [
            Vec3.slice(name="linear", index=np.s_[:3]),
            Vec3.slice(name="angular", index=np.s_[3:]),
        ],
    )
    twist = Twist(np.arange(6))

    IPython.embed()


if __name__ == "__main__":
    main()
