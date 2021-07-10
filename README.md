# namedarray

`namedarray` is a small experiment in Python metaprogramming inspired by
Python's
[namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple)
and projects like [xarray](https://github.com/pydata/xarray).

The goal of the project is to enable the easy creation of wrapper classes
around numpy ndarrays that also provide named attributes for easy
interpretability. Instead of doing this on the dimensional level, as in
`xarray`, I experimented with it on the level of arbitrary slices.

For example, suppose we want to represent a vector in three-dimensional space.
We can do:
```python
from namedarray import namedarray

# create the class using the namedarray factory function
# this class wraps an array of shape (3,) and labels indices (0, 1, 2) as
# (x, y, z)
Vec3 = namedarray(
    name="Vec3",
    shape=(3,),
    slice_map={
        "x": 0,
        "y": 1,
        "z": 2,
    },
)

# initialize with an array
v = Vec3([0, 1, 2])

# I can easily get the array back using
v.array  # array([0, 1, 3])

# but we can also access each attribute individually
v.x  # 0
v.y  # 1
v.z  # 2

# and set them, too
v.x = 3
v.array  # array([3, 1, 2])
```

I imagine this being useful when working with array-based APIs, but you want to
make the code more readable. Since the wrapped array is not copied, you can
easily wrap an array that is being processed elsewhere in the program and just
use the wrapper for inspection.

Wrapper classes can also nest recursively. For example, suppose we want to
represent the spatial twist, which is composed of a 3-component linear
component and a 3-component angular component. Continuing from the code above,
we can do:
```python
# create the twist, where we've also specified that the components should also
# be returned as Vec3 objects
Twist = namedarray(
    name="Twist",
    shape=(6,),
    slice_map={
        "linear": Vec3.slice(np.s_[:3]),
        "angular": Vec3.slice(np.s_[3:]),
    },
)

# again, initialize with an array
twist = Twist([0, 1, 2, 3, 4, 5])

# each component is a Vec3
twist.linear  # Vec3([0, 1, 2])

# so we can also use Vec3's attributes
twist.linear.z  # 2

# but these attributes all just point slices of the same array!
```

I was also motivated by cumbersome nature of trying to use arrays with
non-array-based classes ([ROS](https://www.ros.org/) message classes are a
notable example for me).

Suppose I was working an API for quaternions that gave me a `Quaternion` class
with attributes `x`, `y`, `z`, and `w`. It has some convenient methods that I'd
like to use, such as, for example, a `to_matrix()` method that computes the
equivalent rotation matrix. If I'm otherwise working with quaternions stored as
4-element arrays, to use this method I need to do something like
```python
# my quaternion
q = np.array([0, 0, 0, 1])

# to use the to_matrix() function, I need to do something like
# it's pretty clear what it meant, but it's also a bit verbose
R = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]).to_matrix()
```

Instead, using `namedarray`:
```python
# setup our Quaternion class
QuatArray = namedarray(
    name="QuatArray",
    shape=(4,),
    slice_map={
        "x": 0,
        "y": 1,
        "z": 2,
        "w": 3,
    },
)

# we don't want to redo the implemention work of the original Quaternion class,
# but as long as the relevant attribute names match, we can just steal its
# method
QuatArray.to_matrix = Quaternion.to_matrix

# our quaternion is now
q = QuatArray([0, 0, 0, 1])

# and I can just do 
R = q.to_matrix()
```

## Install
Install with pip:
```
pip install git+https://github.com/adamheins/namedarray.git#egg=namedarray
```

## Develop
This project uses [poetry](https://python-poetry.org/) for dependency
management.

Tests use [pytest](https://docs.pytest.org). Just run `pytest` in the root
directory of the repo to run the tests.
