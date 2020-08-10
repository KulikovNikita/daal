/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cmath>
#include <cstdint>

#ifdef ONEAPI_DAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::backend::primitives {

enum class unary_operation : int { identity = 0, square = 1, abs = 2 };

template <typename Float, unary_operation Op = unary_operation::identity>
struct unary_functor {
    inline Float operator()(Float arg) const;
};

enum class binary_operation : int { min = 0, max = 1, sum = 2, mul = 3 };

template <typename Float, binary_operation Op>
struct binary_functor {
    constexpr static inline Float init_value = static_cast<Float>(NAN);
    inline Float operator()(Float a, Float b) const;
};

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <unary_operation UnOp, binary_operation BinOp, typename Float, bool IsRowMajorLayout>
struct reducer_singlepass_kernel;

}

template <unary_operation UnOp,
          binary_operation BinOp,
          typename Float,
          bool IsRowMajorLayout = true>
struct reducer_singlepass {
public:
    typedef impl::reducer_singlepass_kernel<UnOp, BinOp, Float, IsRowMajorLayout> kernel_t;
    reducer_singlepass(cl::sycl::queue& q);
    cl::sycl::event operator()(array<Float> input,
                               array<Float> output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors,
                               std::int64_t work_items_per_group);
    cl::sycl::event operator()(array<Float> input,
                               array<Float> output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors);
    cl::sycl::event operator()(const Float* input,
                               Float* output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors,
                               std::int64_t work_items_per_group);
    cl::sycl::event operator()(const Float* input,
                               Float* output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors);

private:
    cl::sycl::queue& _q;
    const std::int64_t max_work_group_size;
};

template <typename Float = float, bool IsRowMajorLayout = true>
using l1_reducer_singlepass =
    reducer_singlepass<unary_operation::abs, binary_operation::sum, Float, IsRowMajorLayout>;
template <typename Float = float, bool IsRowMajorLayout = true>
using l2_reducer_singlepass =
    reducer_singlepass<unary_operation::square, binary_operation::sum, Float, IsRowMajorLayout>;
template <typename Float = float, bool IsRowMajorLayout = true>
using linf_reducer_singlepass =
    reducer_singlepass<unary_operation::abs, binary_operation::max, Float, IsRowMajorLayout>;
template <typename Float = float, bool IsRowMajorLayout = true>
using mean_reducer_singlepass =
    reducer_singlepass<unary_operation::identity, binary_operation::sum, Float, IsRowMajorLayout>;
template <typename Float = float, bool IsRowMajorLayout = true>
using geomean_reducer_singlepass =
    reducer_singlepass<unary_operation::identity, binary_operation::mul, Float, IsRowMajorLayout>;

#endif

} // namespace oneapi::dal::backend::primitives