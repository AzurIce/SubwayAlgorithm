import requests

res = requests.get('https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace', headers={
    'x-api-key': 'wfaW5qsRVD3CjWqaCecgz5w3PswW11Dxa3iv3PAl'
})

# with open('./data.zip', 'wb') as f:
#     f.write(res.content)
# print(res.content)

# from gtfslite import GTFS

# gtfs = GTFS.load_zip('./data.zip')
# print(gtfs.summary())

# import proto_pb2

# data = proto_pb2.FeedMessage()
import gtfs_realtime_pb2

data = gtfs_realtime_pb2.FeedMessage()

data.ParseFromString(res.content)
print(data)
