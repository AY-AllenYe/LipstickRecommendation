import sys
import datetime

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.current_line = ""

    def write(self, message):
        self.terminal.write(message)
        # 累积消息直到遇到换行符
        self.current_line += message
        if '\n' in message:
            lines = self.current_line.split('\n')
            # 处理所有完整的行（最后一段可能是不完整的）
            for line in lines[:-1]:
                if line:  # 忽略空行
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log.write(f"[{timestamp}] {line}\n")
            # 保留最后一段不完整的行
            self.current_line = lines[-1]

    def flush(self):
        # 确保程序退出时写入剩余内容
        if self.current_line:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log.write(f"[{timestamp}] {self.current_line}\n")
            self.current_line = ""
        self.log.flush()

    def __del__(self):
        self.flush()
        self.log.close()


# # 重定向标准输出
# sys.stdout = Logger()