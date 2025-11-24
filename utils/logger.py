import sys
import datetime

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.current_line = ""

    def write(self, message):
        self.terminal.write(message)
        self.current_line += message
        if '\n' in message:
            lines = self.current_line.split('\n')
            for line in lines[:-1]:
                if line:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log.write(f"[{timestamp}] {line}\n")
            self.current_line = lines[-1]

    def flush(self):
        if self.current_line:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log.write(f"[{timestamp}] {self.current_line}\n")
            self.current_line = ""
        self.log.flush()

    def __del__(self):
        self.flush()
        self.log.close()

# sys.stdout = Logger()