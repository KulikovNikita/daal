/* file: SplitForwardInput.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * @defgroup split_forward Forward Split Layer
 * @brief Contains classes for the forward split layer
 * @ingroup split
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.split;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPLIT__SPLITFORWARDINPUT"></a>
 * @brief %Input object for the forward split layer
 */
public class SplitForwardInput extends com.intel.daal.algorithms.neural_networks.layers.ForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public SplitForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */