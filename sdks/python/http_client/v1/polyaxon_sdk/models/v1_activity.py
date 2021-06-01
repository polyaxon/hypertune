#!/usr/bin/python
#
# Copyright 2018-2021 Polyaxon, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8

"""
    Polyaxon SDKs and REST API specification.

    Polyaxon SDKs and REST API specification.  # noqa: E501

    The version of the OpenAPI document: 1.9.4
    Contact: contact@polyaxon.com
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from polyaxon_sdk.configuration import Configuration


class V1Activity(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'actor': 'str',
        'owner': 'str',
        'created_at': 'datetime',
        'event_action': 'str',
        'event_subject': 'str',
        'object_name': 'str',
        'object_uuid': 'str',
        'object_parent': 'str'
    }

    attribute_map = {
        'actor': 'actor',
        'owner': 'owner',
        'created_at': 'created_at',
        'event_action': 'event_action',
        'event_subject': 'event_subject',
        'object_name': 'object_name',
        'object_uuid': 'object_uuid',
        'object_parent': 'object_parent'
    }

    def __init__(self, actor=None, owner=None, created_at=None, event_action=None, event_subject=None, object_name=None, object_uuid=None, object_parent=None, local_vars_configuration=None):  # noqa: E501
        """V1Activity - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._actor = None
        self._owner = None
        self._created_at = None
        self._event_action = None
        self._event_subject = None
        self._object_name = None
        self._object_uuid = None
        self._object_parent = None
        self.discriminator = None

        if actor is not None:
            self.actor = actor
        if owner is not None:
            self.owner = owner
        if created_at is not None:
            self.created_at = created_at
        if event_action is not None:
            self.event_action = event_action
        if event_subject is not None:
            self.event_subject = event_subject
        if object_name is not None:
            self.object_name = object_name
        if object_uuid is not None:
            self.object_uuid = object_uuid
        if object_parent is not None:
            self.object_parent = object_parent

    @property
    def actor(self):
        """Gets the actor of this V1Activity.  # noqa: E501


        :return: The actor of this V1Activity.  # noqa: E501
        :rtype: str
        """
        return self._actor

    @actor.setter
    def actor(self, actor):
        """Sets the actor of this V1Activity.


        :param actor: The actor of this V1Activity.  # noqa: E501
        :type: str
        """

        self._actor = actor

    @property
    def owner(self):
        """Gets the owner of this V1Activity.  # noqa: E501


        :return: The owner of this V1Activity.  # noqa: E501
        :rtype: str
        """
        return self._owner

    @owner.setter
    def owner(self, owner):
        """Sets the owner of this V1Activity.


        :param owner: The owner of this V1Activity.  # noqa: E501
        :type: str
        """

        self._owner = owner

    @property
    def created_at(self):
        """Gets the created_at of this V1Activity.  # noqa: E501


        :return: The created_at of this V1Activity.  # noqa: E501
        :rtype: datetime
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at):
        """Sets the created_at of this V1Activity.


        :param created_at: The created_at of this V1Activity.  # noqa: E501
        :type: datetime
        """

        self._created_at = created_at

    @property
    def event_action(self):
        """Gets the event_action of this V1Activity.  # noqa: E501


        :return: The event_action of this V1Activity.  # noqa: E501
        :rtype: str
        """
        return self._event_action

    @event_action.setter
    def event_action(self, event_action):
        """Sets the event_action of this V1Activity.


        :param event_action: The event_action of this V1Activity.  # noqa: E501
        :type: str
        """

        self._event_action = event_action

    @property
    def event_subject(self):
        """Gets the event_subject of this V1Activity.  # noqa: E501


        :return: The event_subject of this V1Activity.  # noqa: E501
        :rtype: str
        """
        return self._event_subject

    @event_subject.setter
    def event_subject(self, event_subject):
        """Sets the event_subject of this V1Activity.


        :param event_subject: The event_subject of this V1Activity.  # noqa: E501
        :type: str
        """

        self._event_subject = event_subject

    @property
    def object_name(self):
        """Gets the object_name of this V1Activity.  # noqa: E501


        :return: The object_name of this V1Activity.  # noqa: E501
        :rtype: str
        """
        return self._object_name

    @object_name.setter
    def object_name(self, object_name):
        """Sets the object_name of this V1Activity.


        :param object_name: The object_name of this V1Activity.  # noqa: E501
        :type: str
        """

        self._object_name = object_name

    @property
    def object_uuid(self):
        """Gets the object_uuid of this V1Activity.  # noqa: E501


        :return: The object_uuid of this V1Activity.  # noqa: E501
        :rtype: str
        """
        return self._object_uuid

    @object_uuid.setter
    def object_uuid(self, object_uuid):
        """Sets the object_uuid of this V1Activity.


        :param object_uuid: The object_uuid of this V1Activity.  # noqa: E501
        :type: str
        """

        self._object_uuid = object_uuid

    @property
    def object_parent(self):
        """Gets the object_parent of this V1Activity.  # noqa: E501


        :return: The object_parent of this V1Activity.  # noqa: E501
        :rtype: str
        """
        return self._object_parent

    @object_parent.setter
    def object_parent(self, object_parent):
        """Sets the object_parent of this V1Activity.


        :param object_parent: The object_parent of this V1Activity.  # noqa: E501
        :type: str
        """

        self._object_parent = object_parent

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1Activity):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, V1Activity):
            return True

        return self.to_dict() != other.to_dict()
