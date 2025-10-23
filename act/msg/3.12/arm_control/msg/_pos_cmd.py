# generated from rosidl_generator_py/resource/_idl.py.em
# with input from arm_control:msg/PosCmd.idl
# generated code does not contain a copyright notice

# This is being done at the module level and not on the instance level to avoid looking
# for the same variable multiple times on each instance. This variable is not supposed to
# change during runtime so it makes sense to only look for it once.
from os import getenv

ros_python_check_fields = getenv('ROS_PYTHON_CHECK_FIELDS', default='')


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'temp_float_data'
# Member 'temp_int_data'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PosCmd(type):
    """Metaclass of message 'PosCmd'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('arm_control')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'arm_control.msg.PosCmd')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__pos_cmd
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__pos_cmd
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__pos_cmd
            cls._TYPE_SUPPORT = module.type_support_msg__msg__pos_cmd
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__pos_cmd

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PosCmd(metaclass=Metaclass_PosCmd):
    """Message class 'PosCmd'."""

    __slots__ = [
        '_x',
        '_y',
        '_z',
        '_roll',
        '_pitch',
        '_yaw',
        '_gripper',
        '_quater_x',
        '_quater_y',
        '_quater_z',
        '_quater_w',
        '_chx',
        '_chy',
        '_chz',
        '_vel_l',
        '_vel_r',
        '_height',
        '_head_pit',
        '_head_yaw',
        '_temp_float_data',
        '_temp_int_data',
        '_mode1',
        '_mode2',
        '_time_count',
        '_check_fields',
    ]

    _fields_and_field_types = {
        'x': 'double',
        'y': 'double',
        'z': 'double',
        'roll': 'double',
        'pitch': 'double',
        'yaw': 'double',
        'gripper': 'double',
        'quater_x': 'double',
        'quater_y': 'double',
        'quater_z': 'double',
        'quater_w': 'double',
        'chx': 'double',
        'chy': 'double',
        'chz': 'double',
        'vel_l': 'double',
        'vel_r': 'double',
        'height': 'double',
        'head_pit': 'double',
        'head_yaw': 'double',
        'temp_float_data': 'double[6]',
        'temp_int_data': 'int32[6]',
        'mode1': 'int32',
        'mode2': 'int32',
        'time_count': 'int32',
    }

    # This attribute is used to store an rosidl_parser.definition variable
    # related to the data type of each of the components the message.
    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int32'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        if 'check_fields' in kwargs:
            self._check_fields = kwargs['check_fields']
        else:
            self._check_fields = ros_python_check_fields == '1'
        if self._check_fields:
            assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
                'Invalid arguments passed to constructor: %s' % \
                ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.x = kwargs.get('x', float())
        self.y = kwargs.get('y', float())
        self.z = kwargs.get('z', float())
        self.roll = kwargs.get('roll', float())
        self.pitch = kwargs.get('pitch', float())
        self.yaw = kwargs.get('yaw', float())
        self.gripper = kwargs.get('gripper', float())
        self.quater_x = kwargs.get('quater_x', float())
        self.quater_y = kwargs.get('quater_y', float())
        self.quater_z = kwargs.get('quater_z', float())
        self.quater_w = kwargs.get('quater_w', float())
        self.chx = kwargs.get('chx', float())
        self.chy = kwargs.get('chy', float())
        self.chz = kwargs.get('chz', float())
        self.vel_l = kwargs.get('vel_l', float())
        self.vel_r = kwargs.get('vel_r', float())
        self.height = kwargs.get('height', float())
        self.head_pit = kwargs.get('head_pit', float())
        self.head_yaw = kwargs.get('head_yaw', float())
        if 'temp_float_data' not in kwargs:
            self.temp_float_data = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.temp_float_data = numpy.array(kwargs.get('temp_float_data'), dtype=numpy.float64)
            assert self.temp_float_data.shape == (6, )
        if 'temp_int_data' not in kwargs:
            self.temp_int_data = numpy.zeros(6, dtype=numpy.int32)
        else:
            self.temp_int_data = numpy.array(kwargs.get('temp_int_data'), dtype=numpy.int32)
            assert self.temp_int_data.shape == (6, )
        self.mode1 = kwargs.get('mode1', int())
        self.mode2 = kwargs.get('mode2', int())
        self.time_count = kwargs.get('time_count', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.get_fields_and_field_types().keys(), self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    if self._check_fields:
                        assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        if self.z != other.z:
            return False
        if self.roll != other.roll:
            return False
        if self.pitch != other.pitch:
            return False
        if self.yaw != other.yaw:
            return False
        if self.gripper != other.gripper:
            return False
        if self.quater_x != other.quater_x:
            return False
        if self.quater_y != other.quater_y:
            return False
        if self.quater_z != other.quater_z:
            return False
        if self.quater_w != other.quater_w:
            return False
        if self.chx != other.chx:
            return False
        if self.chy != other.chy:
            return False
        if self.chz != other.chz:
            return False
        if self.vel_l != other.vel_l:
            return False
        if self.vel_r != other.vel_r:
            return False
        if self.height != other.height:
            return False
        if self.head_pit != other.head_pit:
            return False
        if self.head_yaw != other.head_yaw:
            return False
        if any(self.temp_float_data != other.temp_float_data):
            return False
        if any(self.temp_int_data != other.temp_int_data):
            return False
        if self.mode1 != other.mode1:
            return False
        if self.mode2 != other.mode2:
            return False
        if self.time_count != other.time_count:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def x(self):
        """Message field 'x'."""
        return self._x

    @x.setter
    def x(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'x' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'x' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._x = value

    @builtins.property
    def y(self):
        """Message field 'y'."""
        return self._y

    @y.setter
    def y(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'y' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'y' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._y = value

    @builtins.property
    def z(self):
        """Message field 'z'."""
        return self._z

    @z.setter
    def z(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._z = value

    @builtins.property
    def roll(self):
        """Message field 'roll'."""
        return self._roll

    @roll.setter
    def roll(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'roll' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'roll' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._roll = value

    @builtins.property
    def pitch(self):
        """Message field 'pitch'."""
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'pitch' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'pitch' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._pitch = value

    @builtins.property
    def yaw(self):
        """Message field 'yaw'."""
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'yaw' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'yaw' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._yaw = value

    @builtins.property
    def gripper(self):
        """Message field 'gripper'."""
        return self._gripper

    @gripper.setter
    def gripper(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'gripper' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'gripper' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._gripper = value

    @builtins.property
    def quater_x(self):
        """Message field 'quater_x'."""
        return self._quater_x

    @quater_x.setter
    def quater_x(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'quater_x' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'quater_x' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._quater_x = value

    @builtins.property
    def quater_y(self):
        """Message field 'quater_y'."""
        return self._quater_y

    @quater_y.setter
    def quater_y(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'quater_y' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'quater_y' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._quater_y = value

    @builtins.property
    def quater_z(self):
        """Message field 'quater_z'."""
        return self._quater_z

    @quater_z.setter
    def quater_z(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'quater_z' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'quater_z' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._quater_z = value

    @builtins.property
    def quater_w(self):
        """Message field 'quater_w'."""
        return self._quater_w

    @quater_w.setter
    def quater_w(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'quater_w' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'quater_w' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._quater_w = value

    @builtins.property
    def chx(self):
        """Message field 'chx'."""
        return self._chx

    @chx.setter
    def chx(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'chx' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'chx' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._chx = value

    @builtins.property
    def chy(self):
        """Message field 'chy'."""
        return self._chy

    @chy.setter
    def chy(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'chy' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'chy' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._chy = value

    @builtins.property
    def chz(self):
        """Message field 'chz'."""
        return self._chz

    @chz.setter
    def chz(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'chz' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'chz' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._chz = value

    @builtins.property
    def vel_l(self):
        """Message field 'vel_l'."""
        return self._vel_l

    @vel_l.setter
    def vel_l(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'vel_l' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vel_l' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vel_l = value

    @builtins.property
    def vel_r(self):
        """Message field 'vel_r'."""
        return self._vel_r

    @vel_r.setter
    def vel_r(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'vel_r' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vel_r' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vel_r = value

    @builtins.property
    def height(self):
        """Message field 'height'."""
        return self._height

    @height.setter
    def height(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'height' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'height' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._height = value

    @builtins.property
    def head_pit(self):
        """Message field 'head_pit'."""
        return self._head_pit

    @head_pit.setter
    def head_pit(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'head_pit' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'head_pit' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._head_pit = value

    @builtins.property
    def head_yaw(self):
        """Message field 'head_yaw'."""
        return self._head_yaw

    @head_yaw.setter
    def head_yaw(self, value):
        if self._check_fields:
            assert \
                isinstance(value, float), \
                "The 'head_yaw' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'head_yaw' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._head_yaw = value

    @builtins.property
    def temp_float_data(self):
        """Message field 'temp_float_data'."""
        return self._temp_float_data

    @temp_float_data.setter
    def temp_float_data(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.float64, \
                    "The 'temp_float_data' numpy.ndarray() must have the dtype of 'numpy.float64'"
                assert value.size == 6, \
                    "The 'temp_float_data' numpy.ndarray() must have a size of 6"
                self._temp_float_data = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'temp_float_data' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._temp_float_data = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def temp_int_data(self):
        """Message field 'temp_int_data'."""
        return self._temp_int_data

    @temp_int_data.setter
    def temp_int_data(self, value):
        if self._check_fields:
            if isinstance(value, numpy.ndarray):
                assert value.dtype == numpy.int32, \
                    "The 'temp_int_data' numpy.ndarray() must have the dtype of 'numpy.int32'"
                assert value.size == 6, \
                    "The 'temp_int_data' numpy.ndarray() must have a size of 6"
                self._temp_int_data = value
                return
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 6 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -2147483648 and val < 2147483648 for val in value)), \
                "The 'temp_int_data' field must be a set or sequence with length 6 and each value of type 'int' and each integer in [-2147483648, 2147483647]"
        self._temp_int_data = numpy.array(value, dtype=numpy.int32)

    @builtins.property
    def mode1(self):
        """Message field 'mode1'."""
        return self._mode1

    @mode1.setter
    def mode1(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'mode1' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'mode1' field must be an integer in [-2147483648, 2147483647]"
        self._mode1 = value

    @builtins.property
    def mode2(self):
        """Message field 'mode2'."""
        return self._mode2

    @mode2.setter
    def mode2(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'mode2' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'mode2' field must be an integer in [-2147483648, 2147483647]"
        self._mode2 = value

    @builtins.property
    def time_count(self):
        """Message field 'time_count'."""
        return self._time_count

    @time_count.setter
    def time_count(self, value):
        if self._check_fields:
            assert \
                isinstance(value, int), \
                "The 'time_count' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'time_count' field must be an integer in [-2147483648, 2147483647]"
        self._time_count = value
