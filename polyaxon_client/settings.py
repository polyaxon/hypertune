# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os


class AuthenticationTypes(object):
    TOKEN = 'Token'
    INTERNAL_TOKEN = 'Internaltoken'
    EPHEMERAL_TOKEN = 'EphemeralToken'


IN_CLUSTER = os.getenv('POLYAXON_IN_CLUSTER', False)
API_HOST = os.getenv('POLYAXON_API_HOST', None)
HTTP_PORT = os.getenv('POLYAXON_HTTP_PORT', None)
WS_PORT = os.getenv('POLYAXON_WS_PORT', None)
USE_HTTPS = os.getenv('POLYAXON_USE_HTTPS', False)
API_HTTP_HOST = os.getenv('POLYAXON_API_HTTP_HOST', None)
API_WS_HOST = os.getenv('POLYAXON_API_WS_HOST', None)
SECRET_USER_TOKEN = os.getenv('POLYAXON_SECRET_USER_TOKEN', None)
SECRET_EPHEMERAL_TOKEN = os.getenv('POLYAXON_SECRET_EPHEMERAL_TOKEN', None)
AUTHENTICATION_TYPE = os.getenv('POLYAXON_AUTHENTICATION_TYPE', AuthenticationTypes.TOKEN)
API_VERSION = os.getenv('POLYAXON_API_VERSION', 'v1')
HASH_LENGTH = os.getenv('POLYAXON_HASH_LENGTH', 12)
INTERNAL_HEADER = os.getenv('POLYAXON_INTERNAL_HEADER', None)
INTERNAL_HEADER_SERVICE = os.getenv('POLYAXON_INTERNAL_HEADER_SERVICE', None)
SCHEMA_RESPONSE = os.getenv('POLYAXON_SCHEMA_RESPONSE', False)

DEFAULT_HTTP_PORT = 80
DEFAULT_HTTPS_PORT = 443
TIMEOUT = os.getenv('POLYAXON_TIMEOUT', 10)
