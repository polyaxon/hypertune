// Copyright 2018-2020 Polyaxon, Inc.
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

/*
 * Polyaxon SDKs and REST API specification.
 * Polyaxon SDKs and REST API specification.
 *
 * The version of the OpenAPI document: 1.0.89
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


package org.openapitools.client.model;

import com.google.gson.TypeAdapter;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.annotations.SerializedName;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import java.io.IOException;
import org.openapitools.client.model.V1ArtifactsMount;
import org.openapitools.client.model.V1ArtifactsType;
import org.openapitools.client.model.V1AuthType;
import org.openapitools.client.model.V1CompiledOperation;
import org.openapitools.client.model.V1ConnectionSchema;
import org.openapitools.client.model.V1ConnectionType;
import org.openapitools.client.model.V1DockerfileType;
import org.openapitools.client.model.V1EarlyStopping;
import org.openapitools.client.model.V1Event;
import org.openapitools.client.model.V1EventType;
import org.openapitools.client.model.V1GcsType;
import org.openapitools.client.model.V1GitType;
import org.openapitools.client.model.V1HpParams;
import org.openapitools.client.model.V1ImageType;
import org.openapitools.client.model.V1K8sResourceType;
import org.openapitools.client.model.V1Operation;
import org.openapitools.client.model.V1OperationCond;
import org.openapitools.client.model.V1Parallel;
import org.openapitools.client.model.V1ParallelKind;
import org.openapitools.client.model.V1PolyaxonInitContainer;
import org.openapitools.client.model.V1PolyaxonSidecarContainer;
import org.openapitools.client.model.V1Reference;
import org.openapitools.client.model.V1RunSchema;
import org.openapitools.client.model.V1S3Type;
import org.openapitools.client.model.V1Schedule;
import org.openapitools.client.model.V1UriType;
import org.openapitools.client.model.V1WasbType;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;


/**
 * Model tests for V1Schemas
 */
public class V1SchemasTest {
    private final V1Schemas model = new V1Schemas();

    /**
     * Model tests for V1Schemas
     */
    @Test
    public void testV1Schemas() {
        // TODO: test V1Schemas
    }

    /**
     * Test the property 'operationCond'
     */
    @Test
    public void operationCondTest() {
        // TODO: test operationCond
    }

    /**
     * Test the property 'earlyStopping'
     */
    @Test
    public void earlyStoppingTest() {
        // TODO: test earlyStopping
    }

    /**
     * Test the property 'parallel'
     */
    @Test
    public void parallelTest() {
        // TODO: test parallel
    }

    /**
     * Test the property 'run'
     */
    @Test
    public void runTest() {
        // TODO: test run
    }

    /**
     * Test the property 'operation'
     */
    @Test
    public void operationTest() {
        // TODO: test operation
    }

    /**
     * Test the property 'compiledOperation'
     */
    @Test
    public void compiledOperationTest() {
        // TODO: test compiledOperation
    }

    /**
     * Test the property 'schedule'
     */
    @Test
    public void scheduleTest() {
        // TODO: test schedule
    }

    /**
     * Test the property 'connectionSchema'
     */
    @Test
    public void connectionSchemaTest() {
        // TODO: test connectionSchema
    }

    /**
     * Test the property 'hpParams'
     */
    @Test
    public void hpParamsTest() {
        // TODO: test hpParams
    }

    /**
     * Test the property 'reference'
     */
    @Test
    public void referenceTest() {
        // TODO: test reference
    }

    /**
     * Test the property 'artifactsMount'
     */
    @Test
    public void artifactsMountTest() {
        // TODO: test artifactsMount
    }

    /**
     * Test the property 'polyaxonSidecarContainer'
     */
    @Test
    public void polyaxonSidecarContainerTest() {
        // TODO: test polyaxonSidecarContainer
    }

    /**
     * Test the property 'polyaxonInitContainer'
     */
    @Test
    public void polyaxonInitContainerTest() {
        // TODO: test polyaxonInitContainer
    }

    /**
     * Test the property 'artifacs'
     */
    @Test
    public void artifacsTest() {
        // TODO: test artifacs
    }

    /**
     * Test the property 'wasb'
     */
    @Test
    public void wasbTest() {
        // TODO: test wasb
    }

    /**
     * Test the property 'gcs'
     */
    @Test
    public void gcsTest() {
        // TODO: test gcs
    }

    /**
     * Test the property 's3'
     */
    @Test
    public void s3Test() {
        // TODO: test s3
    }

    /**
     * Test the property 'autg'
     */
    @Test
    public void autgTest() {
        // TODO: test autg
    }

    /**
     * Test the property 'dockerfile'
     */
    @Test
    public void dockerfileTest() {
        // TODO: test dockerfile
    }

    /**
     * Test the property 'git'
     */
    @Test
    public void gitTest() {
        // TODO: test git
    }

    /**
     * Test the property 'uri'
     */
    @Test
    public void uriTest() {
        // TODO: test uri
    }

    /**
     * Test the property 'k8sResource'
     */
    @Test
    public void k8sResourceTest() {
        // TODO: test k8sResource
    }

    /**
     * Test the property 'connection'
     */
    @Test
    public void connectionTest() {
        // TODO: test connection
    }

    /**
     * Test the property 'image'
     */
    @Test
    public void imageTest() {
        // TODO: test image
    }

    /**
     * Test the property 'eventType'
     */
    @Test
    public void eventTypeTest() {
        // TODO: test eventType
    }

    /**
     * Test the property 'event'
     */
    @Test
    public void eventTest() {
        // TODO: test event
    }

    /**
     * Test the property 'parallelKind'
     */
    @Test
    public void parallelKindTest() {
        // TODO: test parallelKind
    }

}
