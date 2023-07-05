import requests

# import BeautifulSoup
from lxml import etree
import csv

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
}
# 请求数据 并获得返回的json数据
with open(r"纽约天气数据.csv", mode="w+", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    # 写入表格的索引
    csv_writer.writerow(["日期", "最高温", "最低温", "天气", "风力风向", "空气质量指数"])
    for year in range(2017, 2024):
        for month in range(1, 13):
            url = (
                "https://tianqi.2345.com/Pc/GetHistory?areaInfo%5BareaId%5D=349727&areaInfo%5BareaType%5D=1&date%5Byear%5D="
                + str(year)
                + "&date%5Bmonth%5D="
                + str(month)
            )
            response = requests.get(url=url, headers=headers)
            json_data = response.json()
            html_data = json_data["data"]
            tree = etree.HTML(html_data)
            tr_list = tree.xpath('//table[@class="history-table"]/tr')
            for tr in tr_list[1:]:
                # td_list=tr.xpath('./td')
                # xpath的索引是从下往上
                d1 = tr.xpath("./td[1]/text()")  # 日期
                d1[0] = d1[0].split(" ")[0]
                d2 = tr.xpath("./td[2]/text()")  # 最高温
                d3 = tr.xpath("./td[3]/text()")  # 最低温
                d4 = tr.xpath("./td[4]/text()")  # 天气
                d5 = tr.xpath("./td[5]/text()")  # 风力风向
                lst = [d1[0], d2[0], d3[0], d4[0], d5[0]]
                # 没得到一行进行一次写入
                csv_writer.writerow(lst)
