# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict, defaultdict
from inspect import signature
from itertools import chain

from typing import Optional, List, Tuple, Dict, Any, Type

from pdf_struct.core.document import TextBlock, Document
from pdf_struct.core.transition_labels import ListAction


def _obtain_inherited_features(cls):
    features = OrderedDict()
    pointer_features = OrderedDict()

    for base_cls in cls.__bases__:
        features = OrderedDict(chain(
            features.items(),
            _obtain_inherited_features(base_cls)[0].items(),
        ))
        pointer_features = OrderedDict(chain(
            pointer_features.items(),
            _obtain_inherited_features(base_cls)[1].items(),
        ))

    if hasattr(cls, '_features'):
        features = OrderedDict(chain(features.items(), cls._features.items()))
        pointer_features = OrderedDict(chain(
            pointer_features.items(), cls._pointer_features.items()))
    return features, pointer_features



class _FeatureRegisterer(type):
    def __init__(cls, name, bases, attrs):
        cls._features, cls._pointer_features = _obtain_inherited_features(cls)
        for key, val in attrs.items():
            properties = getattr(val, '_prop', None)
            if properties is not None:
                name = properties['name']
                if name in cls._features:
                    raise TypeError(f'Duplicate feature name "{name}"')
                if properties['type'] == 'feature':
                    cls._features[name] = {'func': val}
                elif properties['type'] == 'pointer_feature':
                    cls._pointer_features[name] = {'func': val}
                else:
                    assert not 'Should not get here'


def feature(name=None):
    def _body(func):
        nonlocal name
        n_params = len(signature(func).parameters)
        if not (5 <= n_params <= 6):
            raise TypeError(
                '@feature() must receive a function with 4 or 5 arguments (excluding self).')
        if name is None:
            name = func.__name__
        if '-' in name:
            raise ValueError(
                f'@feature name should not include "-" ({name})')

        def _new_func(self, tb1, tb2, tb3, tb4, states):
            if n_params == 5:
                return func(self, tb1, tb2, tb3, tb4)
            else:
                return func(self, tb1, tb2, tb3, tb4, states)

        _new_func._prop = {'name': name, 'type': 'feature'}
        return _new_func
    return _body


def single_input_feature(targets: List[int], name=None):
    if len(set(targets)) != len(targets) or len({0, 1, 2, 3} | set(targets)) != 4:
        raise ValueError('targets must be a list consisting from 0, 1, 2 or 3.')
    def _body(func):
        nonlocal name
        n_params = len(signature(func).parameters)
        if n_params != 2:
            raise TypeError(
                '@single_input_feature() must receive a function with one argument (excluding self).')
        if name is None:
            name = func.__name__
        if '-' in name:
            raise ValueError(
                f'@single_input_feature name should not include "-" ({name})')

        def _new_func(self, *tbs):
            features = dict()
            for i in targets:
                response = func(self, tbs[i])
                if isinstance(response, (int, float, bool)):
                    features[str(i + 1)] = response
                elif isinstance(response, (tuple, list)):
                    features.update({
                        f'{j}_{i}': val for j, val in enumerate(response)
                    })
                elif isinstance(response, dict):
                    if 'states' in response:
                        raise ValueError('@single_input_feature should not return states')
                    features.update({
                        f'{n}_{i}': val for n, val in response.items()
                    })
                else:
                    raise ValueError(
                        'Functions decorated with @single_input_feature must return one of int, float, bool, '
                        f'tuple, list or dict, but "{name}" returned {type(response)}.')
            return features

        _new_func._prop = {'name': name, 'type': 'feature'}
        return _new_func
    return _body


def pairwise_feature(pairs: List[Tuple[int, int]], name=None):
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError('pairs must not include duplicate')
    for pair in pairs:
        if len({0, 1, 2, 3} | set(pair)) != 4 or pair[0] == pair[1]:
            raise ValueError(
                'pairs must be a list of tuples consisting of 0, 1, 2 or 3.')
    def _body(func):
        nonlocal name
        n_params = len(signature(func).parameters)
        if n_params != 3:
            raise TypeError(
                '@pairwise_feature must receive a function with two arguments (excluding self).')
        if name is None:
            name = func.__name__
        if '-' in name:
            raise ValueError(
                f'@pairwise_feature name should not include "-" ({name})')

        def _new_func(self, *tbs):
            features = dict()
            for pair0, pair1 in pairs:
                response = func(self, tbs[pair0], tbs[pair1])
                if isinstance(response, (int, float, bool)):
                    features[f'{pair0 + 1}_{pair1 + 1}'] = response
                elif isinstance(response, (tuple, list)):
                    features.update({
                        f'{j}_{pair0 + 1}_{pair1 + 1}': val for j, val in enumerate(response)
                    })
                elif isinstance(response, dict):
                    if 'states' in response:
                        raise ValueError('@pairwise_feature should not return states')
                    features.update({
                        f'{n}_{pair0 + 1}_{pair1 + 1}': val for n, val in response.items()
                    })
                else:
                    raise ValueError(
                        'Functions decorated with @pairwise_feature must return '
                        'one of int, float, bool, tuple, list or dict, but '
                        f'"{name}" returned {type(response)}.')
            return features

        _new_func._prop = {'name': name, 'type': 'feature'}
        return _new_func
    return _body


def pointer_feature(name=None):
    def _body(func):
        nonlocal name
        n_params = len(signature(func).parameters)
        if n_params != 5:
            raise TypeError(
                '@pointer_feature() must receive a function with 4 arguments (excluding self).')
        if name is None:
            name = func.__name__
        if name in {'transition'}:
            raise ValueError(
                f'The name "{name}" is reserved for @pointer_feature')
        if '-' in name:
            raise ValueError(
                f'@pointer_feature name should not include "-" ({name})')
        func._prop = {'name': name, 'type': 'pointer_feature'}
        return func
    return _body


class BaseFeatureExtractor(object, metaclass=_FeatureRegisterer):
    @staticmethod
    def _parse_feature_response(feature_name, response, state_enabled: bool):
        states = dict()
        if isinstance(response, (int, float, bool)):
            features = {feature_name: float(response)}
        elif isinstance(response, (tuple, list)):
            features = {
                f'{feature_name}_{i}': float(val) for i, val in enumerate(response)
            }
        elif isinstance(response, dict):
            if state_enabled and 'states' in response:
                states[feature_name] = response.pop('states')
            features = dict()
            for name, val in sorted(response.items(), key=lambda k_v: k_v[0]):
                if '-' in name:
                    raise ValueError(
                        '@feature or @pointer_feature name should not include "-" '
                        f'("{name}" in "{feature_name}")')
                features[f'{feature_name}_{name}'] = float(val)
        else:
            raise ValueError(
                'Functions decorated with @feature must return one of int, float, bool, '
                f'tuple, list or dict, but "{feature_name}" returned {type(response)}.')
        if state_enabled:
            return features, states
        else:
            return features

    def extract_features(self, tb1, tb2, tb3, tb4, states
                         ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        features = defaultdict(dict)
        new_states = dict()
        for feature_name, feature_dict in self._features.items():
            ret = feature_dict['func'](self, tb1, tb2, tb3, tb4, states.get(feature_name))
            features_single, states_single = self._parse_feature_response(feature_name, ret, True)
            new_states.update(states_single)
            feature_names_list = sorted(features_single.keys())
            if 'feature_list' in feature_dict:
                if feature_dict['feature_list'] != feature_names_list:
                    raise ValueError(
                        f'Features returned from "{feature_name}" are incosistent with '
                        f'previous runs. ({feature_dict["feature_list"]} vs. {feature_names_list})')
            else:
                self._features[feature_name]['feature_list'] = feature_names_list
            features[feature_name] = features_single
        return dict(features), new_states

    def extract_features_all(self, text_blocks: List[TextBlock], actions: Optional[List[ListAction]]
                             ) -> Dict[str, Dict[str, List[float]]]:
        states = dict()
        features = defaultdict(lambda: defaultdict(list))
        for i in range(len(text_blocks)):
            tb1 = text_blocks[i - 1] if i != 0 else None
            tb2 = text_blocks[i]
            if actions is None or actions[i] == ListAction.ELIMINATE:
                tb3 = text_blocks[i + 1] if i + 1 < len(text_blocks) else None
                tb4 = text_blocks[i + 2] if i + 2 < len(text_blocks) else None
            else:
                tb3 = None
                for j in range(i + 1, len(text_blocks)):
                    if actions[j] != ListAction.ELIMINATE:
                        tb3 = text_blocks[j]
                        break
                tb4 = text_blocks[j + 1] if j + 1 < len(text_blocks) else None
            _features, states = self.extract_features(tb1, tb2, tb3, tb4, states)
            for group_name, feature_group in _features.items():
                for feature_name, feature_val in feature_group.items():
                    features[group_name][feature_name].append(feature_val)
        return {group_name: dict(feature_group) for group_name, feature_group in features.items()}

    def extract_pointer_features(self, text_blocks: List[TextBlock],
                                 list_actions: List[ListAction], i: int, j: int
                                 ) -> Dict[str, Dict[str, float]]:
        # extract features for classifying whether j-th pointer (which
        # determines level at (j+1)-th line) should point at i-th line
        assert 0 <= i < j < len(text_blocks)

        n_downs = len([a for a in list_actions[j:i:-1] if a == ListAction.DOWN])
        n_ups = len([a for a in list_actions[j:i:-1] if a == ListAction.UP])

        for k in range(i, 0, -1):
            if list_actions[k - 1] in {ListAction.UP, ListAction.DOWN,
                                       ListAction.SAME_LEVEL}:
                head_tb = text_blocks[k]
                break
        else:
            head_tb = text_blocks[0]

        if j + 1 >= len(text_blocks):
            tb1, tb2, tb3 = text_blocks[i], text_blocks[j], None
        else:
            tb1, tb2, tb3 = text_blocks[i], text_blocks[j], text_blocks[j+1]
        # note that these feature names are hardcoded to pointer_feature decorator
        features = {
            'transition': {
                'n_downs': n_downs,
                'n_ups': n_ups,
                'n_ups_downs_difference': n_ups - n_downs
            }
        }
        for feature_name, feature_dict in self._pointer_features.items():
            ret = feature_dict['func'](self, head_tb, tb1, tb2, tb3)
            features_single = self._parse_feature_response(feature_name, ret, False)
            feature_names_list = sorted(features_single.keys())
            if 'feature_list' in feature_dict:
                if feature_dict['feature_list'] != feature_names_list:
                    raise ValueError(
                        f'Pointer features returned from "{feature_name}" are incosistent with '
                        f'previous runs. ({feature_dict["feature_list"]} vs. {feature_names_list})')
            else:
                self._pointer_features[feature_name]['feature_list'] = feature_names_list
            features[feature_name] = features_single
        return features

    def extract_pointer_features_all(self, text_blocks, labels, pointers
                                     ) -> Tuple[Dict[str, Dict[str, float]], List[Tuple[int, int]]]:
        features = defaultdict(lambda: defaultdict(list))
        pairs = []
        for j, p in enumerate(pointers):
            if p is not None:
                assert p >= 0
                for i in range(j):
                    if labels[i] == ListAction.DOWN:
                        feat = self.extract_pointer_features(
                            text_blocks, labels[:j], i, j)
                        for group_name, feature_group in feat.items():
                            for feature_name, feature_val in feature_group.items():
                                features[group_name][feature_name].append(feature_val)
                        pairs.append((i, j))
        features = {group_name: dict(feature_group)
                    for group_name, feature_group in features.items()}
        return features, pairs

    @classmethod
    def append_features_to_document(cls, document: Document) -> Document:
        feature_extractor = cls(document.text_blocks)
        feats_test = feature_extractor.extract_features_all(document.text_blocks, None)
        if document.labels is None:
            # prediction
            feats, pointer_feats, pointer_candidates = None, None, None
        else:
            feats = feature_extractor.extract_features_all(document.text_blocks, document.labels)
            pointer_feats, pointer_candidates = feature_extractor.extract_pointer_features_all(
                document.text_blocks, document.labels, document.pointers)
        document.set_features(
            feats, feats_test, pointer_feats, pointer_candidates, feature_extractor
        )
        return document
