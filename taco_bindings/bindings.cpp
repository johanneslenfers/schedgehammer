#include "taco/index_notation/index_notation.h"
#include <Python.h>
#include <chrono>
#include <iostream>
#include <taco.h>
#include <variant>
#include <vector>

using namespace taco;

typedef struct {
    PyObject_HEAD IndexStmt stmt;
    std::vector<IndexVar> vars;
    TensorBase output_tensor;
} ScheduleEnvInternal;

void spmv(ScheduleEnvInternal *se) {
    std::default_random_engine gen(0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    // Predeclare the storage formats that the inputs and output will be stored
    // as. To define a format, you must specify whether each dimension is dense
    // or sparse and (optionally) the order in which dimensions should be
    // stored. The formats declared below correspond to compressed sparse row
    // (csr) and dense vector (dv).
    Format csr({Dense, Sparse});
    Format dv({Dense});

    // Load a sparse matrix from file (stored in the Matrix Market format) and
    // store it as a compressed sparse row matrix. Matrices correspond to
    // order-2 tensors in taco. The matrix in this example can be downloaded
    // from: https://www.cise.ufl.edu/research/sparse/MM/Boeing/pwtk.tar.gz
    Tensor<double> A = read("pwtk/pwtk.mtx", csr);

    // Generate a random dense vector and store it in the dense vector format.
    // Vectors correspond to order-1 tensors in taco.
    Tensor<double> x({A.getDimension(1)}, dv);
    for (int i = 0; i < x.getDimension(0); ++i) {
        x.insert({i}, unif(gen));
    }
    x.pack();

    // Generate another random dense vetor and store it in the dense vector
    // format..
    Tensor<double> z({A.getDimension(0)}, dv);
    for (int i = 0; i < z.getDimension(0); ++i) {
        z.insert({i}, unif(gen));
    }
    z.pack();

    // Declare and initializing the scaling factors in the SpMV computation.
    // Scalars correspond to order-0 tensors in taco.
    Tensor<double> alpha(42.0);
    Tensor<double> beta(33.0);

    // Declare the output matrix to be a sparse matrix with the same dimensions
    // as input matrix B, to be also stored as a doubly compressed sparse row
    // matrix.
    Tensor<double> y({A.getDimension(0)}, dv);
    // Define the SpMV computation using index notation.
    IndexVar i("i"), j("j");
    y(i) = alpha() * (A(i, j) * x(j)) + beta() * z(i);

    se->stmt = y.getAssignment().concretize();
    se->output_tensor = y;
}

extern "C" {

static PyObject *ScheduleEnv_init_spmv(ScheduleEnvInternal *self,
                                       PyObject *Py_UNUSED(ignored)) {
    spmv(self);
    self->vars = self->stmt.getIndexVars();

    Py_RETURN_NONE;
}

static PyObject *ScheduleEnv_get_initial_axes(ScheduleEnvInternal *self,
                                              PyObject *Py_UNUSED(ignored)) {
    auto initial = self->stmt.getIndexVars();

    PyObject *list = PyList_New(0);

    for (auto var : initial) {
        PyList_Append(list, Py_BuildValue("s", var.getName().c_str()));
    }

    return list;
}

static PyObject *ScheduleEnv_get_vars(ScheduleEnvInternal *self,
                                      PyObject *Py_UNUSED(ignored)) {
    PyObject *list = PyList_New(0);

    for (auto var : self->vars) {
        PyList_Append(list, Py_BuildValue("s", var.getName().c_str()));
    }

    return list;
}

static PyObject *ScheduleEnv_split(ScheduleEnvInternal *self, PyObject *args) {
    const char *original, *first, *second;
    int factor;

    if (!PyArg_ParseTuple(args, "sssi", &original, &first, &second, &factor))
        return NULL;

    for (auto var : self->vars) {
        if (var.getName() == original) {
            IndexVar a{std::string(first)}, b{std::string(second)};

            self->stmt.split(var, a, a, factor);
            self->vars.push_back(a);
            self->vars.push_back(b);
        }
    }

    Py_RETURN_NONE;
}

static PyObject *ScheduleEnv_execute(ScheduleEnvInternal *self,
                                     PyObject *Py_UNUSED(ignored)) {
    self->output_tensor.compile(self->stmt);
    self->output_tensor.assemble();

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    self->output_tensor.compute();

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();

    write("y.tns", self->output_tensor);

    return PyLong_FromLong(duration);
}

static PyMethodDef ScheduleEnv_methods[] = {
    {"init_spmv", (PyCFunction)ScheduleEnv_init_spmv, METH_NOARGS,
     "initialize environment with the SPMV computation"},
    {"get_initial_axes", (PyCFunction)ScheduleEnv_get_initial_axes, METH_NOARGS,
     "get initial axes"},
    {"get_vars", (PyCFunction)ScheduleEnv_get_vars, METH_NOARGS,
     "get current index variables"},
    {"split", (PyCFunction)ScheduleEnv_split, METH_VARARGS, "split statement"},
    {"execute", (PyCFunction)ScheduleEnv_execute, METH_NOARGS,
     "execute statement"},
    {NULL} /* Sentinel */
};

static PyTypeObject ScheduleEnv = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name =
        "taco_bindings.ScheduleEnv",
    .tp_basicsize = sizeof(ScheduleEnvInternal),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("ScheduleEnvironment"),
    .tp_methods = ScheduleEnv_methods,
    .tp_new = PyType_GenericNew,
};

static PyObject *testitest(PyObject *Py_UNUSED(ignored_self),
                           PyObject *Py_UNUSED(ignored_args)) {
    Py_RETURN_NONE;
}

static PyMethodDef functions[] = {{"testitest", testitest, METH_NOARGS, NULL},
                                  {NULL, NULL, 0, NULL}};

static struct PyModuleDef bindingsModule = {
    PyModuleDef_HEAD_INIT, "taco_bindings", NULL, -1, functions};

PyMODINIT_FUNC PyInit_taco_bindings(void) {
    PyObject *m = PyModule_Create(&bindingsModule);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&ScheduleEnv) < 0)
        return NULL;

    if (PyModule_AddObjectRef(m, "ScheduleEnv", (PyObject *)&ScheduleEnv) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
}
