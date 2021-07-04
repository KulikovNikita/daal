/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/primitives/reduction/common.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"
#include "oneapi/dal/backend/primitives/super_accumulator/super_accumulator.hpp"

namespace oneapi::dal::backend::primitives {

namespace detail::reduction_rm_cw_super_accum_narrow {

template <typename UnaryOp, int f, int b>
class reduction_kernel {
    using super_accums = super_accumulators<float, false>;

public:
    constexpr static inline std::int64_t rows_per_block = f * b;
    constexpr static inline float zero = 0.f;
    constexpr static inline int folding = f;
    constexpr static inline int block = b;

    reduction_kernel(std::int32_t stride,
                     std::int64_t height,
                     const float* data,
                     std::int64_t* bins,
                     const UnaryOp& unary)
            : stride_(stride),
              height_(height),
              data_(data),
              bins_(bins),
              unary_(unary) {}

    void operator()(sycl::nd_item<2> it) const {
        float acc = zero;
        //
        const std::int32_t bid = it.get_group(1);
        const std::int32_t lid = it.get_local_id(1);
        const std::int32_t hid = it.get_global_id(0);
        //
        // Reduction Section
        //
        const std::int64_t srid = bid * rows_per_block + lid;
        for (std::int32_t i = 0; i < block; ++i) {
            // Current dataset row
            const std::int64_t rid = srid + i * folding;
            // Check for row and col to be in dataset
            const bool handle = rid < height_;
            // Access to the value in row-major order
            // All arithmetics should work in std::int64_t
            const auto& val = data_[hid + rid * stride_];
            acc += handle ? unary_(val) : zero;
        }
        //
        // Super counter Section
        //
        bins_.add(acc, hid);
    }

private:
    const std::int32_t stride_;
    const std::int64_t height_;
    const float* const data_;
    const super_accums bins_;
    const UnaryOp unary_;
};

template <typename UnaryOp, int folding, int block_size, typename = std::enable_if_t<folding != 0>>
inline sycl::event reduction_impl(sycl::queue& queue,
                           const float* data,
                           std::int64_t width,
                           std::int64_t stride,
                           std::int64_t height,
                           std::int64_t* bins,
                           const UnaryOp& unary = {},
                           const std::vector<sycl::event>& deps = {}) {
    using kernel_t = reduction_kernel<UnaryOp, folding, block_size>;
    const auto max_wg = device_max_wg_size(queue);
    const auto cfolding = max_wg / width;
    if (cfolding == folding) {
        constexpr int rpb = kernel_t::rows_per_block;
        const auto n_blocks = (height / rpb) + bool(height % rpb);
        ONEDAL_ASSERT((n_blocks * folding * block) >= height);
        ONEDAL_ASSERT((width * folding) <= max_wg);
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for<kernel_t>(make_multiple_nd_range_2d({ width, folding * n_blocks }, { width, folding }),
                                     kernel_t(dal::detail::integral_cast<std::int32_t>(stride),
                                              height,
                                              data,
                                              bins,
                                              unary));
        });
    }
    if constexpr (folding > 1) {
        return reduction_impl<UnaryOp, folding - 1, block_size>(queue,
                                                                data,
                                                                width,
                                                                stride,
                                                                height,
                                                                bins,
                                                                unary,
                                                                deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event();
}

class finalize_kernel {
    using super_accums = super_accumulators<float, false>;

public:
    finalize_kernel(float* results_, std::int64_t* bins_) : results(results_), all_bins(bins_) {}
    void operator()(sycl::id<1> idx) const {
        results[idx] = all_bins.finalize(idx[0]);
    }

private:
    float* const results;
    const super_accums all_bins;
};

inline sycl::event finalization(sycl::queue& queue,
                         float* results,
                         std::int64_t width,
                         std::int64_t* bins,
                         const std::vector<sycl::event>& deps = {}) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(make_range_1d(width), finalize_kernel(results, bins));
    });
}

} // namespace detail::reduction_rm_cw_super_accum_narrow

template <typename Float, typename BinaryOp, typename UnaryOp>
reduction_rm_cw_super_accum_narrow<Float, BinaryOp, UnaryOp>::reduction_rm_cw_super_accum_narrow(
    sycl::queue& q)
        : q_(q) {}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_narrow<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    std::int64_t* bins,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    using namespace detail::reduction_rm_cw_super_accum_narrow;
    auto reduction_event = reduction_impl<UnaryOp, max_folding, block_size>(q_,
                                                                            input,
                                                                            width,
                                                                            stride,
                                                                            height,
                                                                            bins,
                                                                            unary,
                                                                            deps);
    return finalization(q_, output, width, bins, { reduction_event });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_narrow<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t* bins,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, bins, binary, unary, deps);
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_narrow<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    std::int64_t stride,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    using super_accums = super_accumulators<float, false>;
    const auto bins_size = width * super_accums::nbins;
    auto* const bins = sycl::malloc_device<std::int64_t>(bins_size, q_);
    std::vector<sycl::event> new_deps(deps.size() + 1);
    new_deps[0] = q_.fill<std::int64_t>(bins, 0ul, bins_size);
    std::copy(deps.cbegin(), deps.cend(), ++(new_deps.begin()));
    auto reduction_event =
        this->operator()(input, output, width, height, stride, bins, binary, unary, new_deps);
    reduction_event.wait_and_throw();
    sycl::free(bins, q_);
    return reduction_event;
}

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduction_rm_cw_super_accum_narrow<Float, BinaryOp, UnaryOp>::operator()(
    const Float* input,
    Float* output,
    std::int64_t width,
    std::int64_t height,
    const BinaryOp& binary,
    const UnaryOp& unary,
    const event_vector& deps) const {
    return this->operator()(input, output, width, height, width, binary, unary, deps);
}

#define INSTANTIATE(F, B, U) template class reduction_rm_cw_super_accum_narrow<F, B, U>;

#define INSTANTIATE_FLOAT(B, U) INSTANTIATE(float, B<float>, U<float>);

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
