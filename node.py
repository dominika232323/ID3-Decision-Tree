class Node:
    def __init__(self, feature=None, children=None):
        self.feature = feature
        self.children = children if children is not None else []
