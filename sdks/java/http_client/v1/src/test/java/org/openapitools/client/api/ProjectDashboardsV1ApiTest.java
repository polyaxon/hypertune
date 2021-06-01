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

/*
 * Polyaxon SDKs and REST API specification.
 * Polyaxon SDKs and REST API specification.
 *
 * The version of the OpenAPI document: 1.9.4
 * Contact: contact@polyaxon.com
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


package org.openapitools.client.api;

import org.openapitools.client.ApiException;
import org.openapitools.client.model.RuntimeError;
import org.openapitools.client.model.V1Dashboard;
import org.openapitools.client.model.V1ListDashboardsResponse;
import org.junit.Test;
import org.junit.Ignore;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * API tests for ProjectDashboardsV1Api
 */
@Ignore
public class ProjectDashboardsV1ApiTest {

    private final ProjectDashboardsV1Api api = new ProjectDashboardsV1Api();

    
    /**
     * Create project dashboard
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void createProjectDashboardTest() throws ApiException {
        String owner = null;
        String project = null;
        V1Dashboard body = null;
        V1Dashboard response = api.createProjectDashboard(owner, project, body);

        // TODO: test validations
    }
    
    /**
     * Delete project dashboard
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void deleteProjectDashboardTest() throws ApiException {
        String owner = null;
        String entity = null;
        String uuid = null;
        api.deleteProjectDashboard(owner, entity, uuid);

        // TODO: test validations
    }
    
    /**
     * Get project dashboard
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void getProjectDashboardTest() throws ApiException {
        String owner = null;
        String entity = null;
        String uuid = null;
        V1Dashboard response = api.getProjectDashboard(owner, entity, uuid);

        // TODO: test validations
    }
    
    /**
     * List project dashboard
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void listProjectDashboardNamesTest() throws ApiException {
        String owner = null;
        String name = null;
        Integer offset = null;
        Integer limit = null;
        String sort = null;
        String query = null;
        String mode = null;
        V1ListDashboardsResponse response = api.listProjectDashboardNames(owner, name, offset, limit, sort, query, mode);

        // TODO: test validations
    }
    
    /**
     * List project dashboards
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void listProjectDashboardsTest() throws ApiException {
        String owner = null;
        String name = null;
        Integer offset = null;
        Integer limit = null;
        String sort = null;
        String query = null;
        String mode = null;
        V1ListDashboardsResponse response = api.listProjectDashboards(owner, name, offset, limit, sort, query, mode);

        // TODO: test validations
    }
    
    /**
     * Patch project dashboard
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void patchProjectDashboardTest() throws ApiException {
        String owner = null;
        String project = null;
        String dashboardUuid = null;
        V1Dashboard body = null;
        V1Dashboard response = api.patchProjectDashboard(owner, project, dashboardUuid, body);

        // TODO: test validations
    }
    
    /**
     * Promote project dashboard
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void promoteProjectDashboardTest() throws ApiException {
        String owner = null;
        String entity = null;
        String uuid = null;
        api.promoteProjectDashboard(owner, entity, uuid);

        // TODO: test validations
    }
    
    /**
     * Update project dashboard
     *
     * 
     *
     * @throws ApiException
     *          if the Api call fails
     */
    @Test
    public void updateProjectDashboardTest() throws ApiException {
        String owner = null;
        String project = null;
        String dashboardUuid = null;
        V1Dashboard body = null;
        V1Dashboard response = api.updateProjectDashboard(owner, project, dashboardUuid, body);

        // TODO: test validations
    }
    
}
