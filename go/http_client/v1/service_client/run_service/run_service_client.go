// Copyright 2019 Polyaxon, Inc.
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

// Code generated by go-swagger; DO NOT EDIT.

package run_service

// This file was generated by the swagger tool.
// Editing this file might prove futile when you re-run the swagger generate command

import (
	"fmt"

	"github.com/go-openapi/runtime"

	strfmt "github.com/go-openapi/strfmt"
)

// New creates a new run service API client.
func New(transport runtime.ClientTransport, formats strfmt.Registry) *Client {
	return &Client{transport: transport, formats: formats}
}

/*
Client for run service API
*/
type Client struct {
	transport runtime.ClientTransport
	formats   strfmt.Registry
}

/*
ArchiveRun archives run
*/
func (a *Client) ArchiveRun(params *ArchiveRunParams, authInfo runtime.ClientAuthInfoWriter) (*ArchiveRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewArchiveRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ArchiveRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/archive",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &ArchiveRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*ArchiveRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for ArchiveRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
BookmarkRun bookmarks run
*/
func (a *Client) BookmarkRun(params *BookmarkRunParams, authInfo runtime.ClientAuthInfoWriter) (*BookmarkRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewBookmarkRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "BookmarkRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/bookmark",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &BookmarkRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*BookmarkRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for BookmarkRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
CopyRun restarts run with copy
*/
func (a *Client) CopyRun(params *CopyRunParams, authInfo runtime.ClientAuthInfoWriter) (*CopyRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewCopyRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "CopyRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/copy",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &CopyRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*CopyRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for CopyRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
CreateRun creates new run
*/
func (a *Client) CreateRun(params *CreateRunParams, authInfo runtime.ClientAuthInfoWriter) (*CreateRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewCreateRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "CreateRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &CreateRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*CreateRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for CreateRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
CreateRunCodeRef gets run code ref
*/
func (a *Client) CreateRunCodeRef(params *CreateRunCodeRefParams, authInfo runtime.ClientAuthInfoWriter) (*CreateRunCodeRefOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewCreateRunCodeRefParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "CreateRunCodeRef",
		Method:             "POST",
		PathPattern:        "/api/v1/{entity.owner}/{entity.project}/runs/{entity.uuid}/coderef",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &CreateRunCodeRefReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*CreateRunCodeRefOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for CreateRunCodeRef: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
CreateRunStatus creates new run status
*/
func (a *Client) CreateRunStatus(params *CreateRunStatusParams, authInfo runtime.ClientAuthInfoWriter) (*CreateRunStatusOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewCreateRunStatusParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "CreateRunStatus",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/statuses",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &CreateRunStatusReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*CreateRunStatusOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for CreateRunStatus: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
DeleteRun deletes run
*/
func (a *Client) DeleteRun(params *DeleteRunParams, authInfo runtime.ClientAuthInfoWriter) (*DeleteRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewDeleteRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "DeleteRun",
		Method:             "DELETE",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &DeleteRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*DeleteRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for DeleteRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
DeleteRuns deletes runs
*/
func (a *Client) DeleteRuns(params *DeleteRunsParams, authInfo runtime.ClientAuthInfoWriter) (*DeleteRunsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewDeleteRunsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "DeleteRuns",
		Method:             "DELETE",
		PathPattern:        "/api/v1/{owner}/{project}/runs/delete",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &DeleteRunsReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*DeleteRunsOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for DeleteRuns: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
GetRun gets run
*/
func (a *Client) GetRun(params *GetRunParams, authInfo runtime.ClientAuthInfoWriter) (*GetRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewGetRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "GetRun",
		Method:             "GET",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &GetRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*GetRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for GetRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
GetRunCodeRefs gets run code ref
*/
func (a *Client) GetRunCodeRefs(params *GetRunCodeRefsParams, authInfo runtime.ClientAuthInfoWriter) (*GetRunCodeRefsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewGetRunCodeRefsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "GetRunCodeRefs",
		Method:             "GET",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/coderef",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &GetRunCodeRefsReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*GetRunCodeRefsOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for GetRunCodeRefs: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
GetRunStatuses gets run status
*/
func (a *Client) GetRunStatuses(params *GetRunStatusesParams, authInfo runtime.ClientAuthInfoWriter) (*GetRunStatusesOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewGetRunStatusesParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "GetRunStatuses",
		Method:             "GET",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/statuses",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &GetRunStatusesReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*GetRunStatusesOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for GetRunStatuses: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
InvalidateRun stops run
*/
func (a *Client) InvalidateRun(params *InvalidateRunParams, authInfo runtime.ClientAuthInfoWriter) (*InvalidateRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewInvalidateRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "InvalidateRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/invalidate",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &InvalidateRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*InvalidateRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for InvalidateRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
InvalidateRuns invalidates runs
*/
func (a *Client) InvalidateRuns(params *InvalidateRunsParams, authInfo runtime.ClientAuthInfoWriter) (*InvalidateRunsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewInvalidateRunsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "InvalidateRuns",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/invalidate",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &InvalidateRunsReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*InvalidateRunsOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for InvalidateRuns: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
ListArchivedRuns lists archived runs for user
*/
func (a *Client) ListArchivedRuns(params *ListArchivedRunsParams, authInfo runtime.ClientAuthInfoWriter) (*ListArchivedRunsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewListArchivedRunsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ListArchivedRuns",
		Method:             "GET",
		PathPattern:        "/api/v1/archives/{user}/runs",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &ListArchivedRunsReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*ListArchivedRunsOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for ListArchivedRuns: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
ListBookmarkedRuns lists bookmarked runs for user
*/
func (a *Client) ListBookmarkedRuns(params *ListBookmarkedRunsParams, authInfo runtime.ClientAuthInfoWriter) (*ListBookmarkedRunsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewListBookmarkedRunsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ListBookmarkedRuns",
		Method:             "GET",
		PathPattern:        "/api/v1/bookmarks/{user}/runs",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &ListBookmarkedRunsReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*ListBookmarkedRunsOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for ListBookmarkedRuns: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
ListRuns lists runs
*/
func (a *Client) ListRuns(params *ListRunsParams, authInfo runtime.ClientAuthInfoWriter) (*ListRunsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewListRunsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ListRuns",
		Method:             "GET",
		PathPattern:        "/api/v1/{owner}/{project}/runs",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &ListRunsReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*ListRunsOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for ListRuns: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
PatchRun patches run
*/
func (a *Client) PatchRun(params *PatchRunParams, authInfo runtime.ClientAuthInfoWriter) (*PatchRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewPatchRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "PatchRun",
		Method:             "PATCH",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{run.uuid}",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &PatchRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*PatchRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for PatchRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
RestartRun restarts run
*/
func (a *Client) RestartRun(params *RestartRunParams, authInfo runtime.ClientAuthInfoWriter) (*RestartRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewRestartRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "RestartRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/restart",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &RestartRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*RestartRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for RestartRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
RestoreRun restores run
*/
func (a *Client) RestoreRun(params *RestoreRunParams, authInfo runtime.ClientAuthInfoWriter) (*RestoreRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewRestoreRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "RestoreRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/restore",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &RestoreRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*RestoreRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for RestoreRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
ResumeRun resumes run
*/
func (a *Client) ResumeRun(params *ResumeRunParams, authInfo runtime.ClientAuthInfoWriter) (*ResumeRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewResumeRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "ResumeRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/resume",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &ResumeRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*ResumeRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for ResumeRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
StartRunTensorboard starts run tensorboard
*/
func (a *Client) StartRunTensorboard(params *StartRunTensorboardParams, authInfo runtime.ClientAuthInfoWriter) (*StartRunTensorboardOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewStartRunTensorboardParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "StartRunTensorboard",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/tensorboard/start",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &StartRunTensorboardReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*StartRunTensorboardOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for StartRunTensorboard: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
StopRun stops run
*/
func (a *Client) StopRun(params *StopRunParams, authInfo runtime.ClientAuthInfoWriter) (*StopRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewStopRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "StopRun",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/stop",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &StopRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*StopRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for StopRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
StopRunTensorboard stops run tensorboard
*/
func (a *Client) StopRunTensorboard(params *StopRunTensorboardParams, authInfo runtime.ClientAuthInfoWriter) (*StopRunTensorboardOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewStopRunTensorboardParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "StopRunTensorboard",
		Method:             "DELETE",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/tensorboard/stop",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &StopRunTensorboardReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*StopRunTensorboardOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for StopRunTensorboard: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
StopRuns stops runs
*/
func (a *Client) StopRuns(params *StopRunsParams, authInfo runtime.ClientAuthInfoWriter) (*StopRunsOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewStopRunsParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "StopRuns",
		Method:             "POST",
		PathPattern:        "/api/v1/{owner}/{project}/runs/stop",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &StopRunsReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*StopRunsOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for StopRuns: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
UnBookmarkRun uns bookmark run
*/
func (a *Client) UnBookmarkRun(params *UnBookmarkRunParams, authInfo runtime.ClientAuthInfoWriter) (*UnBookmarkRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewUnBookmarkRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "UnBookmarkRun",
		Method:             "DELETE",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{uuid}/unbookmark",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &UnBookmarkRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*UnBookmarkRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for UnBookmarkRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

/*
UpdateRun updates run
*/
func (a *Client) UpdateRun(params *UpdateRunParams, authInfo runtime.ClientAuthInfoWriter) (*UpdateRunOK, error) {
	// TODO: Validate the params before sending
	if params == nil {
		params = NewUpdateRunParams()
	}

	result, err := a.transport.Submit(&runtime.ClientOperation{
		ID:                 "UpdateRun",
		Method:             "PUT",
		PathPattern:        "/api/v1/{owner}/{project}/runs/{run.uuid}",
		ProducesMediaTypes: []string{"application/json"},
		ConsumesMediaTypes: []string{"application/json"},
		Schemes:            []string{"http", "https", "ws", "wss"},
		Params:             params,
		Reader:             &UpdateRunReader{formats: a.formats},
		AuthInfo:           authInfo,
		Context:            params.Context,
		Client:             params.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	success, ok := result.(*UpdateRunOK)
	if ok {
		return success, nil
	}
	// unexpected success response
	// safeguard: normally, absent a default response, unknown success responses return an error above: so this is a codegen issue
	msg := fmt.Sprintf("unexpected success response for UpdateRun: API contract not enforced by server. Client expected to get an error, but got: %T", result)
	panic(msg)
}

// SetTransport changes the transport on the client
func (a *Client) SetTransport(transport runtime.ClientTransport) {
	a.transport = transport
}
