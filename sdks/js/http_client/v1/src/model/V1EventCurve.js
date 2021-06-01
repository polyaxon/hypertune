// Copyright 2018-2021 Polyaxon, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Polyaxon SDKs and REST API specification.
 * Polyaxon SDKs and REST API specification.
 *
 * The version of the OpenAPI document: 1.9.4
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 *
 */

import ApiClient from '../ApiClient';
import V1EventCurveKind from './V1EventCurveKind';

/**
 * The V1EventCurve model module.
 * @module model/V1EventCurve
 * @version 1.9.4
 */
class V1EventCurve {
    /**
     * Constructs a new <code>V1EventCurve</code>.
     * @alias module:model/V1EventCurve
     */
    constructor() { 
        
        V1EventCurve.initialize(this);
    }

    /**
     * Initializes the fields of this object.
     * This method is used by the constructors of any subclasses, in order to implement multiple inheritance (mix-ins).
     * Only for internal use.
     */
    static initialize(obj) { 
    }

    /**
     * Constructs a <code>V1EventCurve</code> from a plain JavaScript object, optionally creating a new instance.
     * Copies all relevant properties from <code>data</code> to <code>obj</code> if supplied or a new instance if not.
     * @param {Object} data The plain JavaScript object bearing properties of interest.
     * @param {module:model/V1EventCurve} obj Optional instance to populate.
     * @return {module:model/V1EventCurve} The populated <code>V1EventCurve</code> instance.
     */
    static constructFromObject(data, obj) {
        if (data) {
            obj = obj || new V1EventCurve();

            if (data.hasOwnProperty('kind')) {
                obj['kind'] = V1EventCurveKind.constructFromObject(data['kind']);
            }
            if (data.hasOwnProperty('x')) {
                obj['x'] = ApiClient.convertToType(data['x'], ['Number']);
            }
            if (data.hasOwnProperty('y')) {
                obj['y'] = ApiClient.convertToType(data['y'], ['Number']);
            }
            if (data.hasOwnProperty('annotation')) {
                obj['annotation'] = ApiClient.convertToType(data['annotation'], 'String');
            }
        }
        return obj;
    }


}

/**
 * @member {module:model/V1EventCurveKind} kind
 */
V1EventCurve.prototype['kind'] = undefined;

/**
 * @member {Array.<Number>} x
 */
V1EventCurve.prototype['x'] = undefined;

/**
 * @member {Array.<Number>} y
 */
V1EventCurve.prototype['y'] = undefined;

/**
 * @member {String} annotation
 */
V1EventCurve.prototype['annotation'] = undefined;






export default V1EventCurve;

