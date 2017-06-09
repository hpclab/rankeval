/*
 * Authors: Mathieu Blondel <mathieu@mblondel.org>
 *          Lars Buitinck <L.J.Buitinck@uva.nl>
 *
 * License: Simple BSD
 *
 * This module implements _load_svmlight_format, a fast and memory efficient
 * function to load the file format originally created for svmlight and now used
 * by many other libraries, including libsvm.
 *
 * The function loads the file directly in a dense sparse matrix without memory
 * copying.  The approach taken is to use 4 C++ vectors (data, indices, indptr
 * and labels) and to incrementally feed them with elements. Ndarrays are then
 * instantiated by PyArray_SimpleNewFromData, i.e., no memory is
 * copied.
 *
 * Since the memory is not allocated by the ndarray, the ndarray doesn't own the
 * memory and thus cannot deallocate it. To automatically deallocate memory, the
 * technique described at http://blog.enthought.com/?p=62 is used. The main idea
 * is to use an additional object that the ndarray does own and that will be
 * responsible for deallocating the memory.
 */


#include <Python.h>
#include <numpy/arrayobject.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


/*
 * A Python object responsible for memory management of our vectors.
 */
template <typename T>
struct VectorOwner {
  // Inherit from the base Python object.
  PyObject_HEAD
  // The vector that VectorOwner is responsible for deallocating.
  std::vector<T> v;
};

/*
 * Deallocator template.
 */
template <typename T>
static void destroy_vector_owner(PyObject *self)
{
  // Note: explicit call to destructor because of placement new in
  // to_1d_array. memory management for VectorOwner is performed by Python.
  // Compiler-generated destructor will release memory from vector member.
  VectorOwner<T> &obj = *reinterpret_cast<VectorOwner<T> *>(self);
  obj.~VectorOwner<T>();

  self->ob_type->tp_free(self);
}

/*
 * Since a template function can't have C linkage,
 * we instantiate the template for the types "int" and "float"
 * in the following two functions. These are used for the tp_dealloc
 * attribute of the vector owner types further below.
 */
extern "C" {
static void destroy_int_vector(PyObject *self)
{
  destroy_vector_owner<int>(self);
}

static void destroy_float_vector(PyObject *self)
{
  destroy_vector_owner<float>(self);
}
}


/*
 * Type objects for above.
 */
static PyTypeObject IntVOwnerType    = { PyObject_HEAD_INIT(NULL) },
                    FloatVOwnerType = { PyObject_HEAD_INIT(NULL) };

/*
 * Set the fields of the owner type objects.
 */
static void init_type_objs()
{
  IntVOwnerType.tp_flags = FloatVOwnerType.tp_flags = Py_TPFLAGS_DEFAULT;
  IntVOwnerType.tp_name  = FloatVOwnerType.tp_name  = "deallocator";
  IntVOwnerType.tp_doc   = FloatVOwnerType.tp_doc   = "deallocator object";
  IntVOwnerType.tp_new   = FloatVOwnerType.tp_new   = PyType_GenericNew;

  IntVOwnerType.tp_basicsize     = sizeof(VectorOwner<int>);
  FloatVOwnerType.tp_basicsize  = sizeof(VectorOwner<float>);
  IntVOwnerType.tp_dealloc       = destroy_int_vector;
  FloatVOwnerType.tp_dealloc    = destroy_float_vector;
}

PyTypeObject &vector_owner_type(int typenum)
{
  switch (typenum) {
    case NPY_INT: return IntVOwnerType;
    case NPY_FLOAT: return FloatVOwnerType;
  }
  throw std::logic_error("invalid argument to vector_owner_type");
}

/*
 * Parsing.
 */

class SyntaxError : public std::runtime_error {
public:
  SyntaxError(std::string const &msg)
   : std::runtime_error(msg + " in SVMlight/libSVM file")
  {
  }
};



/*
 * Convert a C++ vector to a 1d-ndarray WITHOUT memory copying.
 * Steals v's contents, leaving it empty.
 * Throws an exception if an error occurs.
 */
template <typename T>
static PyObject *to_1d_array(std::vector<T> &v, int typenum)
{
  npy_intp dims[1] = {(npy_intp) v.size()};

  // A C++ vector's elements are guaranteed to be in a contiguous array.
  PyObject *arr = PyArray_SimpleNewFromData(1, dims, typenum, &v[0]);

  try {
    if (!arr)
      throw std::bad_alloc();

    VectorOwner<T> *owner = PyObject_New(VectorOwner<T>,
                                         &vector_owner_type(typenum));
    if (!owner)
      throw std::bad_alloc();

    // Transfer ownership of v's contents to the VectorOwner.
    // Note: placement new.
    new (&owner->v) std::vector<T>();
    owner->v.swap(v);

    PyArray_BASE(arr) = (PyObject *)owner;

    return arr;

  } catch (std::exception const &e) {
    // Let's assume the Python exception is already set correctly.
    Py_XDECREF(arr);
    throw;
  }
}


static PyObject *to_dense(std::vector<float> &data,
                        std::vector<float> &labels,
                        std::vector<int> &qids)
{
  // We could do with a smart pointer to Python objects here.
  std::exception const *exc = 0;
  PyObject *data_arr = 0,
           *qids_arr = 0,
           *labels_arr = 0,
           *ret_tuple = 0;

  try {
    data_arr     = to_1d_array(data, NPY_FLOAT);
    qids_arr     = to_1d_array(qids, NPY_INT);
    labels_arr   = to_1d_array(labels, NPY_FLOAT);

    ret_tuple = Py_BuildValue("OOO",
                              data_arr, labels_arr, qids_arr);

  } catch (std::exception const &e) {
    exc = &e;
  }

  // Py_BuildValue increases the reference count of each array,
  // so we need to decrease it before returning the tuple,
  // regardless of error status.
  Py_XDECREF(data_arr);
  Py_XDECREF(qids_arr);
  Py_XDECREF(labels_arr);

  if (exc)
    throw *exc;

  return ret_tuple;
}


/*
 * Parse single line. Throws exception on failure.
 */
int parse_line(const std::string &line,
                std::vector<float> &data,
                std::vector<float> &labels,
                std::vector<int> &qids,
                int &last_qid
                )
{
  if (line.length() == 0)
    throw SyntaxError("empty line");

  if (line[0] == '#')
    return last_qid;

  // FIXME: we shouldn't be parsing line-by-line.
  // Also, we might catch more syntax errors with failbit.
  size_t hashIdx = line.find('#');
  std::istringstream in(line.substr(0, hashIdx));
  in.exceptions(std::ios::badbit);

    //printf("%s\n",line.substr(0,hashIdx).c_str());
  float y;
  if (!(in >> y)) {
    throw SyntaxError("non-numeric or missing label");
  }
  labels.push_back(y);

  std::string qidNonsense;
  if (!(in >> qidNonsense)) {
    throw SyntaxError("Missing qid label");
  }

  char c;
  double x;
  unsigned idx;
  int idx_row=0;


  if (sscanf(qidNonsense.c_str(), "qid:%u", &idx) != 1) {
    if(sscanf(qidNonsense.c_str(), "%u%c%lf", &idx, &c, &x) == 3) {
        data.push_back(x);
    }
    else {
      throw SyntaxError(std::string("expected ':', got '") + c + "'");
    }
  }
  else {
    if (last_qid == 0){
        last_qid = idx;
        qids.push_back(idx_row);
    }
    if (int(idx) != last_qid){
        idx_row = labels.size()-1;
        qids.push_back(idx_row);
        last_qid = int(idx);
    }
  }

  while (in >> idx >> c >> x) {
    if (c != ':')
      throw SyntaxError(std::string("expected ':', got '") + c + "'");
    data.push_back(x);
  }

  return last_qid;
}

/*
 * Parse entire file. Throws exception on failure.
 */
void parse_file(char const *file_path,
                size_t buffer_size,
                std::vector<float> &data,
                std::vector<float> &labels,
                std::vector<int> &qids)
{
  std::vector<char> buffer(buffer_size);

  std::ifstream file_stream;
  file_stream.exceptions(std::ios::badbit);
  file_stream.rdbuf()->pubsetbuf(&buffer[0], buffer_size);
  file_stream.open(file_path);

  if (!file_stream)
    throw std::ios_base::failure("File doesn't exist!");

  int last_qid = 0;
  int new_qid;
  std::string line;
  while (std::getline(file_stream, line)) {
    new_qid = parse_line(line, data, labels, qids, last_qid);
    last_qid = new_qid;
  }

  /*
  * test is the dataset has qids! if yes -> add SENTINEL
  */
  if (qids.size()!=0){
    qids.push_back(labels.size());
  }
}


static const char load_svmlight_file_doc[] =
  "Load file in svmlight format and return a dense matrix.";

extern "C" {
static PyObject *load_svmlight_file(PyObject *self, PyObject *args)
{

  try {
    // Read function arguments.
    char const *file_path;
    int buffer_mb;

    if (!PyArg_ParseTuple(args, "si", &file_path, &buffer_mb))
      return 0;

    buffer_mb = std::max(buffer_mb, 1);
    size_t buffer_size = buffer_mb * 1024 * 1024;

    std::vector<float> data, labels;
    std::vector<int> qids;
    parse_file(file_path, buffer_size, data, labels, qids);
    return to_dense(data, labels, qids);

  } catch (SyntaxError const &e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return 0;
  } catch (std::bad_alloc const &e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
    return 0;
  } catch (std::ios_base::failure const &e) {
    PyErr_SetString(PyExc_IOError, e.what());
    return 0;
  } catch (std::exception const &e) {
    std::string msg("error in SVMlight/libSVM reader: ");
    msg += e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    return 0;
  }
}
}


static const char dump_svmlight_file_doc[] =
  "Dump dense matrix to a file in svmlight format.";

extern "C" {
static PyObject *dump_svmlight_file(PyObject *self, PyObject *args)
{
  try {
    // Read function arguments.
    char const *file_path;
    PyArrayObject *data_array, *label_array;
    PyObject *query_ids_array;
    int zero_based;

    if (!PyArg_ParseTuple(args,
                          "sO!O!O!i",
                          &file_path,
                          &PyArray_Type, &data_array,
                          &PyArray_Type, &label_array,
                          &PyList_Type,  &query_ids_array,
                          &zero_based))
      return 0;

    int n_samples  = label_array->dimensions[0]; //todo: check -1
    float *data   = (float*) data_array->data;
    float *y      = (float*) label_array->data;
    int n_features = data_array->dimensions[0] / n_samples;

    std::ofstream fout;
    fout.open(file_path, std::ofstream::out);

    float* data_pointer = data;
    for (int i=0; i < n_samples; i++) {
      if (PyList_Size(query_ids_array) != 0) {
        PyObject* pIntObj = PyList_GetItem(query_ids_array, i);
        long qid = PyLong_AsLong(pIntObj);
        fout << y[i] << " qid:" << qid << " ";
      }
      else {
        fout << y[i] << " ";
      }

      for (int jj=0; jj < n_features; jj++) {
        fout << (zero_based ? jj : jj+1) << ":" << data_pointer[jj] << " ";
      }
      data_pointer += n_features;
      fout << std::endl;
    }

    fout.close();

    Py_INCREF(Py_None);
    return Py_None;

  } catch (std::exception const &e) {
    std::string msg("error in SVMlight/libSVM writer: ");
    msg += e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    return 0;
  }
}
}

/*
 * Python module setup.
 */

static PyMethodDef svmlight_format_methods[] = {
  {"_load_svmlight_file", load_svmlight_file,
    METH_VARARGS, load_svmlight_file_doc},

  {"_dump_svmlight_file", dump_svmlight_file,
    METH_VARARGS, dump_svmlight_file_doc},

  {NULL, NULL, 0, NULL}
};

static const char svmlight_format_doc[] =
  "Loader/Writer for svmlight / libsvm datasets - C++ helper routines";

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__svmlight_loader(void)
{
  _import_array();

  init_type_objs();
  if (PyType_Ready(&FloatVOwnerType) < 0
   || PyType_Ready(&IntVOwnerType)    < 0)
#if PY_MAJOR_VERSION >= 3
    return NULL;
#else
	return;
#endif

    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_svmlight_loader",     /* m_name */
        svmlight_format_doc,  /* m_doc */
        -1,                  /* m_size */
        svmlight_format_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
    return PyModule_Create(&moduledef);
}
#else
extern "C" {
PyMODINIT_FUNC init_svmlight_format(void)
{
  _import_array();

  init_type_objs();
  if (PyType_Ready(&FloatVOwnerType) < 0
   || PyType_Ready(&IntVOwnerType)    < 0)
    return;

  Py_InitModule3("_svmlight_format",
                 svmlight_format_methods,
                 svmlight_format_doc);
}
}
#endif
