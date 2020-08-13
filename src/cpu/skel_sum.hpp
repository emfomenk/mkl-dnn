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

#ifndef CPU_SKEL_SUM_HPP
#define CPU_SKEL_SUM_HPP

#include "common/engine.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder_pd.hpp"

#include "cpu/cpu_sum_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct skel_resource_t : public resource_t {
    skel_resource_t() = default;

    status_t configure() {
        // Unused here, but could be used to configure resource's objects
        mutable_object_.reset(new float);
        return status::success;
    }

    float &get_mutable_object() const { return *mutable_object_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(skel_resource_t);

private:
    // Could be a pointer to a mutable descriptor or buffer. For the latter we
    // typically use scratchpad, but for illustration purposes that was the
    // simplest thing to implement.
    std::unique_ptr<float> mutable_object_;
};

struct skel_sum_t : public primitive_t {
    struct pd_t : public cpu_sum_pd_t {
        using cpu_sum_pd_t::cpu_sum_pd_t;

        DECLARE_SUM_PD_T("skel:any", skel_sum_t);

        status_t init(engine_t *engine) {
            bool ok = cpu_sum_pd_t::init(engine) == status::success;
            if (!ok) return status::unimplemented;

            // very strict check to only handle {1,f32} + {1,f32} -> {1,f32}
            memory_desc_t pattern_md;
            dim_t one = 1;
            dnnl_memory_desc_init_by_tag(
                    &pattern_md, 1, &one, data_type::f32, format_tag::x);
            ok = n_inputs() == 2 && *src_md(0) == pattern_md
                    && *src_md(1) == pattern_md && *dst_md() == pattern_md
                    && scales_[0] == 1 && scales_[1] == 1;
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    skel_sum_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override { return status::success; }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<skel_resource_t>();
        if (!r) return status::out_of_memory;

        // Now we can configure the resource as we like. We can pass pd()
        // that contains the information about a problem, or something more
        // specific, say, pd()->jcp_.
        r->configure();
        mapper.add(this, std::move(r));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;

        auto src0 = CTX_IN_MEM(const float *, DNNL_ARG_MULTIPLE_SRC + 0);
        auto src1 = CTX_IN_MEM(const float *, DNNL_ARG_MULTIPLE_SRC + 1);
        auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

        // Retrieve configured resource object at execution
        auto *resoruce = ctx.get_resource_mapper()->get<skel_resource_t>(this);
        float &mutable_object = resoruce->get_mutable_object();

        // Note, this makes the primitive not thread safe, but this should be
        // oK, as it is already the case by default.
        // See DNNL_ENABLE_CONCURRENT_EXEC in cmake/options.cmake.
        mutable_object = src0[0] + src1[0];
        dst[0] = mutable_object;

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
