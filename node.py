class Node:
    def __init__(self, feature, children=None):
        self.feature = feature
        self.children = children if children is not None else []

    def add_child(self, child):
        self.children.append(child)
