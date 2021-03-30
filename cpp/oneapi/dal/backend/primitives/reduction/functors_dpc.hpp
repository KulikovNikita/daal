/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

namespace oneapi::dal::backend::primitives {

namespace unary{

template <typename T>
struct identity_impl {
    T operator()(const T& arg) const {
        return arg;
    }
};

template <typename T>
struct square_impl {
    T operator()(const T& arg) const {
        return (arg * arg);
    }
};

template <typename T>
struct abs_impl {
    T operator()(const T& arg) const {
        return std::abs(arg);
    }
};

struct identity;

struct square;

struct abs;

#ifdef ONEDAL_DATA_PARALLEL

template <typename T>
struct abs_sycl_impl {
    T operator()(const T& arg) const {
        return sycl::fabs(arg);
    }
};

template<typename T, typename Op>
struct native_dpc {
    using type = void;
};

template<typename T>
struct native_dpc<T, identity> {
    using type = identity_impl<T>;
};

template<typename T>
struct native_dpc<T, square> {
    using type = square_impl<T>;
};

template<typename T>
struct native_dpc<T, abs> {
    using type = abs_sycl_impl<T>;
};

template<typename T, typename Op>
using native_dpc_t = typename native_dpc<T, Op>::type;

template<typename T, typename Op>
constexpr inline auto native_dpc_v = native_dpc_t<T, Op>{};

#endif

template<typename T, typename Op>
struct native_host {
    using type = void;
};

template<typename T>
struct native_host<T, identity> {
    using type = identity_impl<T>;
};

template<typename T>
struct native_host<T, square> {
    using type = square_impl<T>;
};

template<typename T>
struct native_host<T, abs> {
    using type = abs_impl<T>;
};

template<typename T, typename Op>
using native_host_t = typename native_host<T, Op>::type;

template<typename T, typename Op>
constexpr inline auto native_host_v = native_host_t<T, Op>{};

}

namespace binary {

template <typename T>
struct sum_impl {
    T operator()(const T& lhs, const T& rhs) const {
        return (lhs + rhs);
    }
};

template <typename T>
struct max_impl {
    T operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs ? rhs : lhs;
    }
};

template <typename T>
struct min_impl {
    T operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs ? lhs : rhs;
    }
};

struct sum;

struct min;

struct max;

#ifdef ONEDAL_DATA_PARALLEL

template<typename T, typename Op>
struct native_dpc {
    using type = void;
};

template<typename T>
struct native_dpc<T, sum> {
    using type = sycl::ONEAPI::plus<T>;
};

template<typename T>
struct native_dpc<T, min> {
    using type = sycl::ONEAPI::minimum<T>;
};

template<typename T>
struct native_dpc<T, max> {
    using type = sycl::ONEAPI::maximum<T>;
};

template<typename T, typename Op>
using native_dpc_t = typename native_dpc<T, Op>::type;

template<typename T, typename Op>
constexpr inline auto native_dpc_v = native_dpc_t<T, Op>{};

#endif

template<typename T, typename Op>
struct native_host {
    using type = void;
};

template<typename T>
struct native_host<T, sum> {
    using type = sum_impl<T>;
};

template<typename T>
struct native_host<T, max> {
    using type = max_impl<T>;
};

template<typename T>
struct native_host<T, min> {
    using type = min_impl<T>;
};

template<typename T, typename Op>
using native_host_t = typename native_host<T, Op>::type;

template<typename T, typename Op>
constexpr inline auto native_host_v = native_host_t<T, Op>{};

template<typename T, typename Op>
struct neutral_value {
    constexpr inline T value = 0; 
};

template<typename T>
struct neutral_value<T, sum> {
    constexpr inline static T value = 0; 
};

template<typename T>
struct neutral_value<T, max> {
    constexpr inline static T value = std::numeric_limits<T>::min(); 
};

template<typename T>
struct neutral_value<T, min> {
    constexpr inline static T value = std::numeric_limits<T>::max(); 
};

constexpr inline auto native_dpc_v = native_value<T, Op>:value;

}

#endif

} // namespace oneapi::dal::backend::primitives
