from subprocess import check_call
from time import sleep
from datetime import datetime

while True:
    hours_date = (datetime.now()).strftime("%Y%m%d%H")[-2:] # 获取小时
    # if int(hours_date)%4==0:
    cmd ='python tcn_predict.py'
    print(hours_date)
    check_call(cmd, shell=True)
    print("...执行结束")
    sleep(3599*4) # 跳过 目标时间段
