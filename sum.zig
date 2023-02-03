const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", "1");
    @cInclude("Python.h");
});

const PyObject = c.PyObject;

const PyModuleDef_Base = extern struct {
    ob_base: PyObject,
    // m_init: ?fn () callconv(.C) [*c]PyObject = null,
    m_init: ?*const fn () callconv(.C) [*c]PyObject = null,
    m_index: c.Py_ssize_t = 0,
    m_copy: [*c]PyObject = null,
};

const PyModuleDef_HEAD_INIT = PyModuleDef_Base {
    .ob_base = PyObject {
        .ob_refcnt = 1,
        .ob_type = null,
    }
};

const PyMethodDef = extern struct {
    ml_name: [*c]const u8 = null,
    ml_meth: c.PyCFunction = null,
    ml_flags: c_int = 0,
    ml_doc: [*c]const u8 = null,
};

const PyModuleDef = extern struct {
    // m_base: c.PyModuleDef_Base,
    m_base: PyModuleDef_Base = PyModuleDef_HEAD_INIT,
    m_name: [*c]const u8,
    m_doc: [*c]const u8 = null,
    m_size: c.Py_ssize_t = -1,
    m_methods: [*]PyMethodDef,
    m_slots: [*c]c.struct_PyModuleDef_Slot = null,
    m_traverse: c.traverseproc = null,
    m_clear: c.inquiry = null,
    m_free: c.freefunc = null,
};

/////////////////////////////////////////////////

pub export fn sum(self: [*]PyObject, args: [*]PyObject) [*c]PyObject {
    var a: c_long = undefined;
    var b: c_long = undefined;
    _ = self;
    if (!(c._PyArg_ParseTuple_SizeT(args, "ll", &a, &b) != 0)) return null;
    return c.PyLong_FromLong((a + b));
}

pub var methods = [_:PyMethodDef{}]PyMethodDef{
    PyMethodDef{
        .ml_name = "sum",
        .ml_meth = @ptrCast(c.PyCFunction, @alignCast(@import("std").meta.alignment(c.PyCFunction), &sum)),
        .ml_flags = @as(c_int, 1),
        .ml_doc = null,
    },
};

pub var zigmodule = PyModuleDef{
    .m_name = "zig_sum",
    .m_methods = &methods,
};

pub export fn PyInit_zig_sum() [*c]c.PyObject {
    return c.PyModule_Create(@ptrCast([*c]c.struct_PyModuleDef, &zigmodule));
}