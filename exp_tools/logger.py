class Logger:
    def __init__(self, name=None, level=1):
        self.name = name
        self.level = level

    def set_name(self, name):
        self.name = name

    def get_header(self):
        return f'[{self.name}]' if self.name else ''

    def header_print(self, msg, *arg, **kwarg):
        msg = str(msg)
        lines = msg.splitlines()
        for i, line in enumerate(lines):
            if i == len(lines)-1:
                print(f"{self.get_header()} {line}", *arg, **kwarg)
            else:
                print(f"{self.get_header()} {line}")

    def log(self, msg, *arg, **kwarg):
        if msg:
            self.header_print(msg, *arg, **kwarg)
        else:
            print()

    def warn(self, msg, *arg, **kwarg):
        self.header_print(msg, *arg, **kwarg)

    def err(self, msg, *arg, **kwarg):
        self.header_print(msg, *arg, **kwarg)

    def fatal(self, msg, *arg, **kwarg):
        self.header_print(msg, *arg, **kwarg)
        exit()

logger = Logger()