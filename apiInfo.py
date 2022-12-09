import pandas as pd
import requests
import pprint
from os import name
import xml.etree.ElementTree as et
import pandas as pd
import bs4
from lxml import html
from urllib.parse import urlencode, quote_plus, unquote

# 강남구, 강동구, 강북구, 강서구, 관악구, 광진구, 구로구, 금천구, 노원구, 도봉구
# 동대문구, 동작구, 마포구, 서대문구, 서초구, 성동구, 성북구, 송파구, 양천구, 영등포구
# 용산구, 은평구, 종로구, 중구, 중랑구
seoul = [11680, 11740, 11305, 11500, 11620, 11215, 11530, 11545, 11350, 11320, 11230, 11590 , 11440, 11410, 11650, 11200, 11290, 11710, 11470, 11560, 11170, 11380, 11110, 11260]
gu = ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구"
      "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구"
      "용산구", "은평구", "종로구", "중구", "중랑구"]

dt_idx = pd.date_range(start = "20160101", end = "20211231").strftime("%Y%m").unique().tolist()

url = 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade'
result = []

for code in seoul:
	for time in dt_idx:
		params ={'serviceKey' : 'p1P0B3ibG2uL6TW+45WMf4a+m7RlOtGZ0EPKaKzNli19nRSFzcNeKTa4b2tN+QV41ftTRUcLNPyTXQDncg5evw==', 'LAWD_CD' : str(code), 'DEAL_YMD' : str(time)}
		response = requests.get(url, params=params)
		content = response.content

		pp = pprint.PrettyPrinter(indent = 4)
		xml_obj = bs4.BeautifulSoup(content,'lxml-xml')
		rows = xml_obj.findAll('item')

		row_list = [] # row value
		name_list = [] # column name
		value_list = [] # data

		# xml data collect
		for i in range(0, len(rows)):
			columns = rows[i].find_all()
			for j in range(0,len(columns)):
				if i == 0:
					name_list.append(columns[j].name)
				value_list.append(columns[j].text)
			row_list.append(value_list)
			value_list=[]

		# to df
		df = pd.DataFrame(row_list, columns=name_list)		
		result.append(df)

data = pd.concat(result, axis = 0).to_csv("seoul.csv", index = None, mode = "w")
