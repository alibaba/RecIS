#include "column-io/py_interface/converter.h"

#include <string>
#include <dlpack.h>
#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include "absl/log/log.h"

#include "column-io/framework/types.h"

namespace column {
namespace py_interface {
template<typename T>
struct Capsule {
  Capsule() = default;
  Capsule(T tensor) : tensor(tensor) {} // tensor , or other obj
  pybind11::capsule ToPyCapsule() {
    return pybind11::capsule(this, [](void *ptr) {
      Capsule *origin = (Capsule *)ptr;
      delete origin;
    });
  }
  T tensor;
};

pybind11::object CastTensor(Tensor tensor) {
  pybind11::object ret;
  if (!tensor.Initialized()) {
    ret = pybind11::none();
  }
  if (!tensor.Shape().IsScalar()) {
    switch (tensor.Type()) {
#define COLUMN_LOCAL_TYPE(DTYPE)                                               \
  case column::DTYPE: {                                                        \
    using real_type = column::EnumToDataType<column::DTYPE>::Type;             \
    size_t element_size = column::SizeOfType(column::DTYPE);                   \
    std::vector<size_t> strides;                                               \
    strides.resize(tensor.Shape().Size());                                     \
    size_t last = element_size;                                                \
    int64_t index = strides.size() - 1;                                        \
    while (index >= 0) {                                                       \
      strides[index] = last;                                                   \
      last *= tensor.Shape().Dims()[index];                                    \
      index--;                                                                 \
    }                                                                          \
    Capsule<Tensor> *capsule = new Capsule<Tensor>(tensor);                    \
    ret = pybind11::array(                                                     \
        pybind11::buffer_info(                                                 \
            tensor.Raw<real_type>(), column::SizeOfType(column::DTYPE),        \
            pybind11::format_descriptor<real_type>::format(),                  \
            tensor.Shape().Size(), tensor.Shape().Dims(), strides),            \
        capsule->ToPyCapsule());                                               \
    break;                                                                     \
  }
      COLUMN_LOCAL_TYPE(kBool);
      COLUMN_LOCAL_TYPE(kInt8);
      COLUMN_LOCAL_TYPE(kInt16);
      COLUMN_LOCAL_TYPE(kInt32);
      COLUMN_LOCAL_TYPE(kInt64);
      COLUMN_LOCAL_TYPE(kUInt8);
      COLUMN_LOCAL_TYPE(kUInt16);
      COLUMN_LOCAL_TYPE(kUInt32);
      COLUMN_LOCAL_TYPE(kUInt64);
      COLUMN_LOCAL_TYPE(kFloat);
      COLUMN_LOCAL_TYPE(kDouble);
#undef COLUMN_LOCAL_TYPE
    case column::kString: {
      // 此处写法不支持Null
      using real_type = column::EnumToDataType<column::kString>::Type;
      auto origin_ptr = tensor.Raw<real_type>();
      size_t max_len = 0;
      for (size_t i = 0; i < tensor.NumElements(); i++) {
        max_len = std::max(max_len, origin_ptr[i].size());
      }
      size_t total_len = max_len * tensor.NumElements() + 1;
      column::Tensor new_tensor({1}, column::kString);
      auto &new_str = new_tensor.Raw<real_type>()[0];
      new_str.resize(total_len);
      auto new_str_ptr = &(new_str[0]);
      for (size_t i = 0; i < tensor.NumElements(); i++) {
        strncpy(new_str_ptr + i * max_len, origin_ptr[i].data(), max_len);
      }
      Capsule<Tensor> *capsule = new Capsule<Tensor>(new_tensor);
      ret = pybind11::array(pybind11::dtype(std::string("S")+std::to_string(max_len)),
                            {tensor.Shape().Dims()}, {max_len}, new_str.data(),
                            capsule->ToPyCapsule());
    }
    }
  } else {
    switch (tensor.Type()) {
#define COLUMN_LOCAL_TYPE(DTYPE)                                               \
  case column::DTYPE: {                                                        \
    ret = pybind11::cast(                                                      \
        tensor.Scalar<column::EnumToDataType<column::DTYPE>::Type>());         \
    break;                                                                     \
  }
      COLUMN_LOCAL_TYPE(kBool);
      COLUMN_LOCAL_TYPE(kInt8);
      COLUMN_LOCAL_TYPE(kInt16);
      COLUMN_LOCAL_TYPE(kInt32);
      COLUMN_LOCAL_TYPE(kInt64);
      COLUMN_LOCAL_TYPE(kUInt8);
      COLUMN_LOCAL_TYPE(kUInt16);
      COLUMN_LOCAL_TYPE(kUInt32);
      COLUMN_LOCAL_TYPE(kUInt64);
      COLUMN_LOCAL_TYPE(kFloat);
      COLUMN_LOCAL_TYPE(kDouble);
    case column::kString: {
      using real_type = column::EnumToDataType<column::kString>::Type;
      ret = pybind11::bytes(tensor.Raw<real_type>()[0].c_str(),
                            tensor.Raw<real_type>()[0].size());
    }
#undef COLUMN_LOCAL_TYPE
    }
  }
  return ret;
}

// Cast A Tensor Element(row) Into Primitive Python Object (e.g. int, float, str, bool.)
inline pybind11::object ConvertToPrimitiveObject(const column::Tensor& tensor, size_t row) {
    if(tensor.IsNull(row)) {
        return pybind11::none();
    }
    switch (tensor.Type()) {
        // TODO: make them template or macro, not repeat
        // MATCH_TYPE_AND_ENUM(int, float, double..., kInt, kFloat, kDouble...);
        case column::kInt8:
            return pybind11::int_(tensor.Raw<int8_t>()[row]);
        case column::kInt16:
            return pybind11::int_(tensor.Raw<int16_t>()[row]);
        case column::kInt32:
            return pybind11::int_(tensor.Raw<int32_t>()[row]);
        case column::kInt64:
            return pybind11::int_(tensor.Raw<int64_t>()[row]);
        case column::kUInt8:
            return pybind11::int_(tensor.Raw<uint8_t>()[row]);
        case column::kUInt16:
            return pybind11::int_(tensor.Raw<uint16_t>()[row]);
        case column::kUInt32:
            return pybind11::int_(tensor.Raw<uint32_t>()[row]);
        case column::kUInt64:
            return pybind11::int_(tensor.Raw<uint64_t>()[row]);
        case column::kFloat:
            return pybind11::float_(tensor.Raw<float>()[row]);
        case column::kDouble:
            return pybind11::float_(tensor.Raw<double>()[row]);
        case column::kBool:
            return pybind11::bool_(tensor.Raw<bool>()[row]);
        case column::kString: {
            return pybind11::str(tensor.Raw<std::string>()[row]);
        }
        default:
            throw std::runtime_error("Unsupported type: " + 
                                   std::to_string(tensor.Type()));
    }
}
// Cast A Tensor Element(row) Into Pure Python-style Object
inline pybind11::object ConvertToPyObject(const column::Tensor& tensor, size_t row) {
    // if(tensor.IsNull(row)) {
    //     return pybind11::none();
    // }
    // 1. 处理基本类型
    // 1.1  0维 (通常不会发生, 意味着batch的一列是标量)
    // 1.2  1维 [标量]
    if( tensor.Shape().IsScalar()) {
        return CastTensor(tensor);
    }
    if( tensor.Shape().Size() <= 1 ) { // dim_ = { row_cnt, }
        return ConvertToPrimitiveObject(tensor, row);
    }
    // 2+维
    // ---------------- ↓↓↓↓ COPIED&EDITED FROM CastTensor ↓↓↓↓ ---------------- //
    switch (tensor.Type()) {
        #define COLUMN_LOCAL_TYPE(DTYPE)                                               \
        case column::DTYPE: {                                                          \
            using real_type = column::EnumToDataType<column::DTYPE>::Type;             \
            size_t element_size = column::SizeOfType(column::DTYPE);                   \
            std::vector<size_t> strides;                                               \
            strides.resize(tensor.Shape().Size());                                     \
            size_t last = element_size;                                                \
            int64_t index = strides.size() - 1;                                        \
            while (index >= 0) {                                                       \
            strides[index] = last;                                                     \
            last *= tensor.Shape().Dims()[index];                                      \
            index--;                                                                   \
            }                                                                          \
            size_t row_stride = last;                                                  \
            strides.erase(strides.begin() );                                           \
            std::vector<std::size_t> dims(tensor.Shape().Dims().begin() + 1, tensor.Shape().Dims().end() );     \
            Capsule<Tensor> *capsule = new Capsule<Tensor>(tensor);                     \
            return pybind11::array(                                                     \
                pybind11::buffer_info(                                                  \
                    tensor.Raw<real_type>(), column::SizeOfType(column::DTYPE),         \
                    pybind11::format_descriptor<real_type>::format(),                   \
                    tensor.Shape().Size() -1, dims, strides),                           \
                capsule->ToPyCapsule());                                                \
        }
        COLUMN_LOCAL_TYPE(kBool);
        COLUMN_LOCAL_TYPE(kInt8);
        COLUMN_LOCAL_TYPE(kInt16);
        COLUMN_LOCAL_TYPE(kInt32);
        COLUMN_LOCAL_TYPE(kInt64);
        COLUMN_LOCAL_TYPE(kUInt8);
        COLUMN_LOCAL_TYPE(kUInt16);
        COLUMN_LOCAL_TYPE(kUInt32);
        COLUMN_LOCAL_TYPE(kUInt64);
        COLUMN_LOCAL_TYPE(kFloat);
        COLUMN_LOCAL_TYPE(kDouble);
    #undef COLUMN_LOCAL_TYPE
        // case column::kString: {
        // using real_type = column::EnumToDataType<column::kString>::Type;
        // if( tensor.Shape().Dims().size() != 2 ) {
        //     // TODO: 支持多层嵌套, 同时解决null问题
        //     throw std::runtime_error("Unsupported multi-dim type: Dim:" + std::to_string(tensor.Shape().Dims().size()) + "*Type:" + std::to_string(tensor.Type()));
        // }
        // auto origin_ptr = tensor.Raw<real_type>();
        // std::vector<std::string> str_vec;
        // pybind11::list temp_list;

        // str_vec.resize(tensor.Shape().Dims()[1]);
        // for(size_t i=0; i<tensor.Shape().Dims()[1]; i++) {
        //     std::string str = origin_ptr[row*tensor.Shape().Dims()[1] + i];
        //     temp_list.append(pybind11::bytes(str));
        // }
        // // get pybind11::array(pybind11::dtype("S"),) from str_vec
        // // 创建Python列表

        // // 转换为numpy array
        // pybind11::array arr = pybind11::array(temp_list);
        // return arr;

        // size_t max_len = 0; /////////////
        case column::kString: {
            using real_type = column::EnumToDataType<column::kString>::Type;
            int64_t row_ele_cnt = 1;
            for (size_t i = 1; i < tensor.Shape().Dims().size(); i++) {
                row_ele_cnt *= tensor.Shape().Dims()[i];
            }
            std::vector<std::size_t> row_dims(tensor.Shape().Dims().begin() + 1, tensor.Shape().Dims().end());

            auto origin_ptr = tensor.Raw<real_type>();
            size_t max_len = 0;
            for (size_t i = row*row_ele_cnt; i < (row+1)*row_ele_cnt; i++) {
                max_len = std::max(max_len, origin_ptr[i].size());
            }

            column::Tensor new_tensor({1}, column::kString);
            auto &new_str = new_tensor.Raw<real_type>()[0];
            new_str.resize(max_len * row_ele_cnt + 1);
            auto new_str_ptr = &(new_str[0]);
            for (size_t i = row*row_ele_cnt; i < (row+1)*row_ele_cnt; i++) {
                strncpy(new_str_ptr + i * max_len, origin_ptr[i].data(), max_len); // DOUBT: why max_len, not origin_ptr[i].size() ?
            }
            Capsule<column::Tensor> *capsule = new Capsule<column::Tensor>(new_tensor);
            // TODO: 这里S*不支持null element. 同时pybind11::array也不能为null. 需要考虑如何从RecordBatch到Tensor时保留空数组和组内空元素的信息
            return pybind11::array(pybind11::dtype(std::string("S")+std::to_string(max_len)),
                                    {row_dims}, {max_len}, new_str.data(),
                                    capsule->ToPyCapsule());
        }
        default:
            return pybind11::none();
    // ---------------- ↑↑↑↑ COPIED&EDITED FROM CastTensor ↑↑↑↑ ---------------- //
    }
}

pybind11::list CastTensorsToPythonTuples(std::vector<Tensor> tensors, std::vector<size_t>& outputs_row_spliter) {
    /*  tensors: {  tensor0, tensor1, tensor2, ... }
                    tensor0: { row0, row1, row2, ... }
        outputs_row_spliter: { col0_tensor_idx_begin, col0_tensor_idx_end, col1_tensor_idx_end, col2_tensor_idx_end, ... }
        ret: {  tuple{ col[0][0], col[1][0], col[2][0], ... },
                tuple{ col[0][1], col[1][1], col[2][1], ... }, ... } 
    */
    // 1. return object generate
    pybind11::list ret;
    if(tensors.empty() || outputs_row_spliter.size() <= 1) {
        LOG(INFO) << "CastTensorsToPythonTuples with empty tensors:" << tensors.size() << " or spliters:"<< outputs_row_spliter.size();
        return ret;
    }
    // tensors.len >= outputs_row_spliter-1,  == happens when no struct type(list,map,etc) in column type
    size_t num_cols = outputs_row_spliter.size() -1;
    size_t num_rows = 0;
    
    // 2. schema check for every column's length
    for(int col=0;col<num_cols; col++) {
        size_t num_col_rows = 0;
        size_t begin = outputs_row_spliter[col];
        size_t end = outputs_row_spliter[col+1];
        if( begin + 1 < end ){ // if no spliter-tensor, pure primary type
            num_col_rows = tensors[begin+1].NumElements() - 1; // assume it's 2ax3bx4c, then tensors[begin+1]is: [ a0begin, a0end, a1end ]
        }else{
            num_col_rows = tensors[begin].NumElements();
        } // (begin + 1 > end) never happens
        if ( num_rows > num_col_rows ) {
            std::ostringstream oss;
            oss << "CastTensorsToPythonTuples prev tensor len: " << num_rows << " unequal current tensor len: " << num_col_rows;
            LOG(ERROR) << oss.str();
            throw std::runtime_error(oss.str());
        }
        num_rows = std::max(num_rows, num_col_rows);
    }

    // 3. convert
    // std::vector<pybind11::tuple> batch(num_rows, pybind11::tuple(num_cols)); // pybind11::tuple is not changable
    std::vector<std::vector<pybind11::object>> batch(num_rows, std::vector<pybind11::object>(num_cols));
    for(int j=0;j<num_cols; j++) {
        size_t begin = outputs_row_spliter[j];
        size_t end = outputs_row_spliter[j+1];
        Tensor& tensor = tensors[begin];
        if( begin + 1 == end ){ // SCALAR
            for(size_t i=0; i<num_rows; i++) {
                batch[i][j] = ConvertToPyObject(tensor, i);
            }
        } else if (begin +2 == end){
            for(size_t i=0; i<num_rows; i++) {
                // pybind11::list obj; // element of batch[i][j]
                int32_t split_start = tensors[end-1].Raw<int32_t>()[i];
                int32_t split_lenth = tensors[end-1].Raw<int32_t>()[i+1] - split_start;
                pybind11::list obj(split_lenth);
                for(int l = 0; l < split_lenth; l++) {
                    obj[l] = ConvertToPyObject(tensor, split_start + l);
                }
                batch[i][j] = obj;
            }
        }else { // TODO: 验证生成的多层结构和tunnel或paiio对比是否正确. 以下内容正确性尚未验证
            LOG(WARNING) << "CastTensorsToPythonTuples with odps array<array<>> type may unsupported yet";
            pybind11::list obj;
            pybind11::list swap_obj;
            // Get leaf layer
            for(size_t i=0; i<tensors[end-1].NumElements()-1; i++) {
                int32_t leaf_split_start = tensors[end-1].Raw<int32_t>()[i];
                int32_t leaf_split_end = tensors[end-1].Raw<int32_t>()[i+1];
                for(int l = leaf_split_start; l < leaf_split_end; l++) {
                    obj.append(ConvertToPyObject(tensor, l)); // TODO: 检查null元素是否统计到 NumElements (dim)中
                }
            }
            // Get Middle&Top layer
            for(size_t layer = end-2; layer >begin; layer--) {
                Tensor& spliter = tensors[layer];
                for(size_t k=0; k<spliter.NumElements()-1; k++) {
                    int32_t layer_split_start = spliter.Raw<int32_t>()[k];
                    int32_t layer_split_end = spliter.Raw<int32_t>()[k+1];
                    for(int l = layer_split_start; l < layer_split_end; l++) {
                        swap_obj.append( obj[l] );
                    }
                    obj = swap_obj;
                    // swap_obj.clear() raise ‘class pybind11::list’ has no member named 'clear' err
                    swap_obj.attr("clear")();
                }
            }
            // Assign Top layer as batch[i][j]
            for(size_t i=0; i<num_rows; i++) {
                batch[i][j] = obj[i];
            }
        }
    }

   // cast vector<vector<py::object>> into py::list<py::tuple<py::object>>
    for (auto& row : batch) {
        pybind11::tuple tup(row.size());
        for (size_t i = 0; i < row.size(); ++i) {
            tup[i] = row[i];
        }
        ret.append(tup);
    }
    return ret;
}

void deleter(DLManagedTensor* tensor) {
  delete[] tensor->dl_tensor.shape;
  delete static_cast<Tensor*>(tensor->manager_ctx);
  delete tensor;
}

pybind11::capsule CreateCapsule(Tensor tensor, const DLDataType& dtype) {
    // DLManagedTensorVersioned* managed = new DLManagedTensorVersioned();
    DLManagedTensor* managed = new DLManagedTensor();
    managed->dl_tensor.data = static_cast<void*>(tensor.mutable_data());
    managed->dl_tensor.device = tensor.Dev();
    managed->dl_tensor.dtype = dtype;
    managed->dl_tensor.ndim = tensor.dims();
    managed->dl_tensor.shape = new int64_t[tensor.dims()];
    memcpy(managed->dl_tensor.shape, tensor.Shape().Dims().data(), sizeof(int64_t) * tensor.dims());
    managed->dl_tensor.strides = nullptr;
    managed->manager_ctx = new Tensor(std::move(tensor));
    managed->deleter = deleter;
    // managed->flags = 0;
    // managed->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};

    return pybind11::capsule(managed, "dltensor", [](PyObject* obj) {
        // DLManagedTensorVersioned* t = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(obj, "dltensor"));
        if (PyCapsule_IsValid(obj, "used_dltensor")) {
          return;  /* Do nothing if the capsule has been consumed. */
        }
        DLManagedTensor* t = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "dltensor"));
        if (t != nullptr) {
            t->deleter(t);
        }
    });
}

pybind11::object CastTensorToDLPack(Tensor tensor) {
  pybind11::object ret;
  if (!tensor.Initialized()) {
    ret = pybind11::none();
  }
  if (!tensor.Shape().IsScalar()) {
    switch (tensor.Type()) {
#define COLUMN_LOCAL_TYPE(DTYPE, ...)                                               \
  case column::DTYPE: {                                                        \
    DLDataType dltype{__VA_ARGS__}; \
    using real_type = column::EnumToDataType<column::DTYPE>::Type;             \
    size_t element_size = column::SizeOfType(column::DTYPE);                   \
    std::vector<size_t> strides;                                               \
    strides.resize(tensor.Shape().Size());                                     \
    size_t last = element_size;                                                \
    int64_t index = strides.size() - 1;                                        \
    while (index >= 0) {                                                       \
      strides[index] = last;                                                   \
      last *= tensor.Shape().Dims()[index];                                    \
      index--;                                                                 \
    }                                                                          \
    ret = CreateCapsule(tensor, dltype);                                   \
    break;                                                                     \
  }
      COLUMN_LOCAL_TYPE(kBool, kDLBool, 8, 1);
      COLUMN_LOCAL_TYPE(kInt8, kDLInt, 8, 1);
      COLUMN_LOCAL_TYPE(kInt16, kDLInt, 16, 1);
      COLUMN_LOCAL_TYPE(kInt32, kDLInt, 32, 1);
      COLUMN_LOCAL_TYPE(kInt64, kDLInt, 64, 1);
      COLUMN_LOCAL_TYPE(kUInt8, kDLUInt, 8, 1);
      COLUMN_LOCAL_TYPE(kUInt16, kDLUInt, 16, 1);
      COLUMN_LOCAL_TYPE(kUInt32, kDLUInt, 32, 1);
      COLUMN_LOCAL_TYPE(kUInt64, kDLUInt, 64, 1);
      COLUMN_LOCAL_TYPE(kFloat, kDLFloat, 32, 1);
      COLUMN_LOCAL_TYPE(kDouble, kDLFloat, 64, 1);
#undef COLUMN_LOCAL_TYPE
    case column::kString: {
      using real_type = column::EnumToDataType<column::kString>::Type;
      auto origin_ptr = tensor.Raw<real_type>();
      size_t max_len = 0;
      for (size_t i = 0; i < tensor.NumElements(); i++) {
        max_len = std::max(max_len, origin_ptr[i].size());
      }
      size_t total_len = max_len * tensor.NumElements() + 1;
      column::Tensor new_tensor({1}, column::kString);
      auto &new_str = new_tensor.Raw<real_type>()[0];
      new_str.resize(total_len);
      auto new_str_ptr = &(new_str[0]);
      for (size_t i = 0; i < tensor.NumElements(); i++) {
        strncpy(new_str_ptr + i * max_len, origin_ptr[i].data(), max_len);
      }
      Capsule<Tensor> *capsule = new Capsule<Tensor>(new_tensor);
      ret = pybind11::array(pybind11::dtype(std::string("S")+std::to_string(max_len)),
                            {tensor.Shape().Dims()}, {max_len}, new_str.data(),
                            capsule->ToPyCapsule());
    }
    }
  } else {
    switch (tensor.Type()) {
#define COLUMN_LOCAL_TYPE(DTYPE)                                               \
  case column::DTYPE: {                                                        \
    ret = pybind11::cast(                                                      \
        tensor.Scalar<column::EnumToDataType<column::DTYPE>::Type>());         \
    break;                                                                     \
  }
      COLUMN_LOCAL_TYPE(kBool);
      COLUMN_LOCAL_TYPE(kInt8);
      COLUMN_LOCAL_TYPE(kInt16);
      COLUMN_LOCAL_TYPE(kInt32);
      COLUMN_LOCAL_TYPE(kInt64);
      COLUMN_LOCAL_TYPE(kUInt8);
      COLUMN_LOCAL_TYPE(kUInt16);
      COLUMN_LOCAL_TYPE(kUInt32);
      COLUMN_LOCAL_TYPE(kUInt64);
      COLUMN_LOCAL_TYPE(kFloat);
      COLUMN_LOCAL_TYPE(kDouble);
    case column::kString: {
      using real_type = column::EnumToDataType<column::kString>::Type;
      ret = pybind11::bytes(tensor.Raw<real_type>()[0].c_str(),
                            tensor.Raw<real_type>()[0].size());
    }
#undef COLUMN_LOCAL_TYPE
    }
  }
  return ret;
}
} // namespace py_interface
} // namespace column
