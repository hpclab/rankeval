/*
 * Authors: Mathieu Blondel <mathieu@mblondel.org>
 *          Lars Buitinck <L.J.Buitinck@uva.nl>
 *          Salvatore Trani <salvatore.trani@isti.cnr.it>
 *
 * License: Simple BSD
 *
 * This module implements _load_svmlight_format, a fast and memory efficient
 * function to load the file format originally created for svmlight and now used
 * by many other libraries, including libsvm.
 *
 * The function loads the file directly in a dense matrix without memory
 * copying.  The approach taken is to use 2 C++ vectors (data, and labels)
 * and to incrementally feed them with elements. If the dataset is sparse,
 * the function will fix the previously loaded instances in order to reflect
 * the new observed column (given 0-value to missing features). Ndarrays are
 * then instantiated by PyArray_SimpleNewFromData, i.e., no memory is
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
#include <iostream>
#include <iomanip>

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
    labels_arr   = to_1d_array(labels, NPY_FLOAT);
    qids_arr     = to_1d_array(qids, NPY_INT);

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
 * Reshape the data array to adjust number of columns
 * (caused by the loading of a sparse matrix)
 */
void reshape_data(std::vector<float> &data,
                  int &old_num_feature,
                  int new_num_feature)
{
  int rows_to_fix = (data.size() - new_num_feature) / old_num_feature;
  int cells_to_add = rows_to_fix * (new_num_feature - old_num_feature);
  int cells_to_add_per_row = cells_to_add / rows_to_fix;

  for (int i=0; i<cells_to_add; ++i)
    data.push_back(0);

  std::vector<float>::iterator it = data.end() - new_num_feature - cells_to_add;
  int move_counter = cells_to_add;
  for (int i=0; i<=rows_to_fix; ++i) {
    int window = (i == 0) ? new_num_feature : old_num_feature;
    if (move_counter > 0)
      std::copy(it, it + window, it + move_counter);

    if (i > 0) {
      for (int j=0; j<cells_to_add_per_row; ++j)
        *(it + move_counter + old_num_feature + j) = 0;
    }

    move_counter -= cells_to_add_per_row;
    it -= old_num_feature;
  }
}


/*
 * Parse single line. Throws exception on failure.
 */
void parse_line(const std::string &line,
                std::vector<float> &data,
                std::vector<float> &labels,
                std::vector<int> &qids,
                int &max_feature)
{
  if (line.length() == 0)
  	throw std::invalid_argument( "empty line" );

  if (line[0] == '#')
    return;

  // FIXME: we shouldn't be parsing line-by-line.
  // Also, we might catch more syntax errors with failbit.
  size_t hashIdx = line.find('#');
  std::istringstream in(line.substr(0, hashIdx));
  in.exceptions(std::ios::badbit);

    //printf("%s\n",line.substr(0,hashIdx).c_str());
  float y;
  if (!(in >> y)) {
  	throw std::invalid_argument( "non-numeric or missing label" );
  }
  labels.push_back(y);

  std::string qidNonsense;
  if (!(in >> qidNonsense)) {
  	throw std::invalid_argument( "Missing qid label" );
  }

  char c;
  double x;
  int idx;
  int next_feature = 1;

  if (sscanf(qidNonsense.c_str(), "qid:%u", &idx) != 1) {
    if(sscanf(qidNonsense.c_str(), "%u%c%lf", &idx, &c, &x) == 3) {
        // Add zeros in empty spaces between next_feature and idx indices  (iff idx > next_feature)
        for (; next_feature < idx; ++next_feature)
          data.push_back(0);
        data.push_back(x);
        ++next_feature;
    } else {
    	throw std::invalid_argument( std::string("expected ':', got '") + c + "'");
    }

  } else {
    qids.push_back(idx);
  }

  while (in >> idx >> c >> x) {
    if (c != ':')
    	throw std::invalid_argument( std::string("expected ':', got '") + c + "'");
    // Add zeros in empty spaces between next_feature and idx indices (iff idx > next_feature)
    for (; next_feature < idx; ++next_feature)
      data.push_back(0);
    data.push_back(x);
    ++next_feature;
  }

  // Add zeros at the end of the row (iff next_feature < max_feature)
  for (; next_feature <= max_feature; ++next_feature)
    data.push_back(0);

  // if the maximum feature read is greater than the maximum feature read since here,
  // it means we have to reshape the dataset to include new columns...
  if (max_feature > 0 && (next_feature - 1) > max_feature) {
    reshape_data(data, max_feature, next_feature - 1);
  }

  max_feature = std::max(next_feature - 1, max_feature);
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

  int max_feature = 0;
  std::string line;
  while (std::getline(file_stream, line)) {
    parse_line(line, data, labels, qids, max_feature);
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
    PyArrayObject *data_array, *label_array, *qids_array;
    int zero_based;

    if (!PyArg_ParseTuple(args,
                          "sO!O!O!i",
                          &file_path,
                          &PyArray_Type, &data_array,
                          &PyArray_Type, &label_array,
                          &PyArray_Type,  &qids_array,
                          &zero_based))
      return 0;

    int n_samples = PyArray_DIM(data_array, 0);
    int n_features = PyArray_DIM(data_array, 1);
    int n_queries = PyArray_DIM(qids_array, 0) - 1;
//    int n_samples  = label_array->dimensions[0];
//    int n_features = data_array->dimensions[0] / n_samples;

    float *data   = (float*) data_array->data;
    float *y      = (float*) label_array->data;
    int   *qids   = (int*) qids_array->data;

    std::ofstream fout;
    fout.precision(8);
    fout.open(file_path, std::ofstream::out);

    float* data_pointer = data;
    for (int i=0; i < n_samples; i++) {
      if (n_queries > 0) {
        int qid = qids[i];
        fout << y[i] << " qid:" << qid << " ";
      } else {
        fout << y[i] << " ";
      }

      for (int jj=0; jj < n_features; ++jj) {
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
