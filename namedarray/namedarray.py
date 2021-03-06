from functools import lru_cache, partial
from dataclasses import dataclass

import numpy as np


@dataclass
class TypedSlice:
    slice_: slice
    cls: type


class NamedArray:
    """Base class for all classes generated by the namedarray function.

    Wraps a numpy array.
    """
    def __init__(self, array=None):
        if array is None:
            self.array = np.zeros(self.shape)
        else:
            self.array = np.array(array, copy=False)
        assert self.array.shape == self.shape

    def __array__(self):
        return self.array

    def __repr__(self):
        return f"{type(self).__name__}({list(self.array)})"

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
        # TODO it would be nice to handle tranlation of field names as well
        # (the names shouldn't have to be the same)
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
    def slice(cls, slice_):
        return TypedSlice(slice_=slice_, cls=cls)



def namedarray(
    typename,
    shape,
    slice_map,
):
    """Returns a new class with names for array slices.

    The new class wraps a numpy ndarray and allows getting and setting
    arbitrary slices by name."""


    cls = type(typename, (NamedArray,), {})
    cls.shape = shape
    cls._names = []

    def fget_typed(self, slice_, cls):
        return cls(self.array[slice_])

    def fget(self, slice_):
        return self.array[slice_]

    def fset(self, value, slice_):
        self.array[slice_] = np.array(value, copy=False)

    # Create the properties on the objects, each of which gets/sets a
    # particular slice of the underlying array
    for name, slice_ in slice_map.items():
        cls._names.append(name)

        setter = partial(fset, slice_=slice_)

        if isinstance(slice_, TypedSlice):
            # if the property gets initialized to another object, we don't want
            # to recreate it each time, so we cache it
            getter = lru_cache(maxsize=None)(
                partial(fget_typed, slice_=slice_.slice_, cls=slice_.cls)
            )
        else:
            getter = partial(fget, slice_=slice_)

        setattr(cls, name, property(fget=getter, fset=setter))

    return cls
