from subprocess import check_call
from time import sleep
from datetime import datetime, timedelta
while True:
    hours_date = (datetime.now()).strftime("%Y%m%d%H")[-2:] # 获取小时
    if int(hours_date)%4==0:
        cmd =  'python predict.py'
        check_call(cmd, shell=True)
        sleep(3600*4) # 跳过 目标时间段
