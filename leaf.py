class Leaf:
    def __int__(self, class_name):
        self._class_name = class_name

    @property
    def class_name(self):
        return self._class_name
