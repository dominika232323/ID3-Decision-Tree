class Node:
    def __init__(self, feature=None, children=None, feature_value_branch=None, class_name=None):
        self._feature = feature
        self._children = children if children is not None else []
        self._feature_value_branch = feature_value_branch
        self._class_name = class_name

    @property
    def feature(self):
        return self._feature

    @property
    def children(self):
        return self._children

    @property
    def feature_value_branch(self):
        return self._feature_value_branch

    @property
    def class_name(self):
        return self._class_name

    def is_leaf(self):
        return self._class_name is not None

    def set_feature(self, feature):
        self._feature = feature

    def add_child(self, child):
        self._children.append(child)

    def set_feature_value_branch(self, feature_value_branch):
        self._feature_value_branch = feature_value_branch

    def set_class_name(self, class_name):
        self._class_name = class_name
