class Queue(object):

    def __init__(self):
        self.nextin = 0
        self.nextout = 0
        self.data = {}

    def append(self, value):
        self.data[self.nextin] = value
        self.nextin += 1

    def pop(self):
        value = self.data.pop(self.nextout)
        self.nextout += 1
        return value

    def is_empty(self):
        return self.nextout == self.nextin