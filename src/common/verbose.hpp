/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#ifndef VERBOSE_HPP
#define VERBOSE_HPP

#include <cinttypes>
#include <mutex>
#include <stdio.h>

#include "c_types_map.hpp"
#include "dnnl_debug.h"
#include "utils.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {

struct verbose_t {
    int level;
};

int get_verbose();
double get_msec();
const char *get_isa_info();

#if !defined(DISABLE_VERBOSE)
#define DNNL_VERBOSE_BUF_LEN 1024
#else
#define DNNL_VERBOSE_BUF_LEN 1
#endif

/** A container for primitive desc verbose string.
 *
 * The buffer is kept in `str` member. All housekeeping is on caller... The
 * structure only helps with the synchronization and lazy copying.
 *
 * Synchronization model:
 *  - Both `is_initialized` and `initialization_flag` are controlled by a caller
 *    code. The `initialization_flag` allows avoiding race conditions.
 *  - Once the info is initialized, the `initialized` flag set to true, and
 *    (in case of primitive desc copy) the `initialization_flag` essentially
 *    becomes redundant.
 */
struct pd_info_t {
    pd_info_t() = default;
    pd_info_t(const pd_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {}
    pd_info_t &operator=(const pd_info_t &rhs) {
        is_initialized_ = rhs.is_initialized_;
        str_ = rhs.str_;
        return *this;
    }

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(const primitive_desc_t *pd);

private:
    std::string str_;

#if defined(DISABLE_VERBOSE)
    bool is_initialized_ = true; // no verbose -> info is always ready
#else
    bool is_initialized_ = false;
#endif
    std::once_flag initialization_flag_;
};

} // namespace impl
} // namespace dnnl

#endif
