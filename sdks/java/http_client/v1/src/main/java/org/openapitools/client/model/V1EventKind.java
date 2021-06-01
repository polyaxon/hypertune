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


package org.openapitools.client.model;

import java.util.Objects;
import java.util.Arrays;
import com.google.gson.annotations.SerializedName;

import java.io.IOException;
import com.google.gson.TypeAdapter;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

/**
 * Gets or Sets v1EventKind
 */
@JsonAdapter(V1EventKind.Adapter.class)
public enum V1EventKind {
  
  RUN_STATUS_CREATED("run_status_created"),
  
  RUN_STATUS_RESUMING("run_status_resuming"),
  
  RUN_STATUS_COMPILED("run_status_compiled"),
  
  RUN_STATUS_ON_SCHEDULE("run_status_on_schedule"),
  
  RUN_STATUS_QUEUED("run_status_queued"),
  
  RUN_STATUS_SCHEDULED("run_status_scheduled"),
  
  RUN_STATUS_STARTING("run_status_starting"),
  
  RUN_STATUS_RUNNING("run_status_running"),
  
  RUN_STATUS_PROCESSING("run_status_processing"),
  
  RUN_STATUS_STOPPING("run_status_stopping"),
  
  RUN_STATUS_FAILED("run_status_failed"),
  
  RUN_STATUS_STOPPED("run_status_stopped"),
  
  RUN_STATUS_SUCCEEDED("run_status_succeeded"),
  
  RUN_STATUS_SKIPPED("run_status_skipped"),
  
  RUN_STATUS_WARNING("run_status_warning"),
  
  RUN_STATUS_UNSCHEDULABLE("run_status_unschedulable"),
  
  RUN_STATUS_UPSTREAM_FAILED("run_status_upstream_failed"),
  
  RUN_STATUS_RETRYING("run_status_retrying"),
  
  RUN_STATUS_UNKNOWN("run_status_unknown"),
  
  RUN_STATUS_DONE("run_status_done"),
  
  RUN_APPROVED_ACTOR("run_approved_actor"),
  
  RUN_INVALIDATED_ACTOR("run_invalidated_actor"),
  
  RUN_NEW_ARTIFACTS("run_new_artifacts"),
  
  CONNECTION_GIT_COMMIT("connection_git_commit"),
  
  CONNECTION_DATASET_VERSION("connection_dataset_version"),
  
  CONNECTION_REGISTRY_IMAGE("connection_registry_image"),
  
  ALERT_INFO("alert_info"),
  
  ALERT_WARNING("alert_warning"),
  
  ALERT_CRITICAL("alert_critical"),
  
  MODEL_VERSION_NEW_METRIC("model_version_new_metric"),
  
  PROJECT_CUSTOM_EVENT("project_custom_event"),
  
  ORG_CUSTOM_EVENT("org_custom_event");

  private String value;

  V1EventKind(String value) {
    this.value = value;
  }

  public String getValue() {
    return value;
  }

  @Override
  public String toString() {
    return String.valueOf(value);
  }

  public static V1EventKind fromValue(String value) {
    for (V1EventKind b : V1EventKind.values()) {
      if (b.value.equals(value)) {
        return b;
      }
    }
    throw new IllegalArgumentException("Unexpected value '" + value + "'");
  }

  public static class Adapter extends TypeAdapter<V1EventKind> {
    @Override
    public void write(final JsonWriter jsonWriter, final V1EventKind enumeration) throws IOException {
      jsonWriter.value(enumeration.getValue());
    }

    @Override
    public V1EventKind read(final JsonReader jsonReader) throws IOException {
      String value = jsonReader.nextString();
      return V1EventKind.fromValue(value);
    }
  }
}

