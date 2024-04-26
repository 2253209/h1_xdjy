from datetime import datetime
import os

class SimpleLogger:
    def __init__(self, path):

        now = datetime.now()
        # 将当前时间格式化为字符串
        formatted_time = now.strftime('%Y-%m-%d_%H:%M:%S')
        filename = f"{path}/log_{formatted_time}.csv"
        if not os.path.exists(path):
            os.mkdir(path)
        self.file = open(filename, "a+")
        print(f"Saving log! Path: {filename}")
        i = 0
        label = 'No'
        for k in range(42):
            if k == 1:
                label = 'sin'
                i = 0
            elif k == 3:
                label = 'cmd'
                i = 0
            elif k == 6:
                label = 'pos'
                i = 0
            elif k == 16:
                label = 'vel'
                i = 0
            elif k == 26:
                label = 'act'
                i = 0
            elif k == 36:
                label = 'omg'
                i = 0
            elif k == 39:
                label = 'eul'
                i = 0
            self.file.write(f'{label}_{i},')
            i += 1

        self.file.write('time,')
        self.file.write('\n')

    def save(self, obs, step, time):
        for row in obs:
            k = 0
            self.file.write('%d, ' % step)
            for index, item in enumerate(row):
                if 26 <= index < 36:
                    self.file.write(' %.4f,' % (item / 4))
                else:
                    self.file.write(' %.4f,' % item)
                k += 1
            self.file.write(' %d,' % int(time * 10 ** 6))
            self.file.write('\n')

    def close(self):
        self.file.close()


