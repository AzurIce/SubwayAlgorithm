## MTA 的 GTFS 解析

- EntitySelector
  - `string agency_id`
  - `string route_id`
  - `int32 route_type`
  - `TripDescriptor trip`
  - `string stop_id`



- VehicleDescriptor 标识



- TranslateString

  - Translation translation

    > Translation 结构如下
    >
    > - string text
    > - string language

```
FeedMessage {
	FeedHeader header {
		string gtfs_realtime_version
		enum Incrementality {
            FULL_DATASET
            DIFFERENTIAL
        }
        Incrementality incrementality = 2 [default = FULL_DATASET];
        timestamp = 3;
        
        // extensions
	}
	FeedEntity entity {
		string id
		bool is_deleted
		TripUpdate trip_update {
			TripDescriptor trip
			VehicleDescriptor vehicle
			StopTimeEvent {
				delay
				time
				uncertainty
				
				// extensions
			}
			StopTimeUpdate {
				stop_sequence
				stop_id
				StopTimeEvent arrival
				StopTimeEvent departure
				enum ScheduleRelationship {
                    SCHEDULED = 0;
                    SKIPPED = 1;
                    NO_DATA = 2;
                }
			}
			StopTimeUpdate stop_time_update
			timestamp
			delay
		}
		VehiclePosition vehicle {
			TripDescriptor trip {
				trip_id
				route_id
				direction_id
				start_time
				start_date
				enum ScheduleRelationship {
                    SCHEDULED
                    ADDED
                    UNSCHEDULED
					CANCELED = 3;
                }
			}
			VehicleDescriptor vehicle {
				id
				label
				license_plate
			}
			Position position {
				latitude
				longitude
				bearing
				odometer
				speed
			}
			current_stop_sequence
			stop_id
			enum VehicleStopStatus {
                INCOMING_AT
                STOPPED_AT
                IN_TRANSIT_TO
            }
            VehicleStopStatus current_status
            timestamp
            enum CongestionLevel {
                UNKNOWN_CONGESTION_LEVEL
                RUNNING_SMOOTHLY
                STOP_AND_GO
                CONGESTION
                SEVERE_CONGESTION
            }
            CongestionLevel congestion_level
            enum OccupancyStatus {
                EMPTY
                MANY_SEATS_AVAILABLE
                FEW_SEATS_AVAILABLE
                STANDING_ROOM_ONLY
                CRUSHED_STANDING_ROOM_ONLY
                FULL
                NOT_ACCEPTING_PASSENGERS
            }
            OccupancyStatus occupancy_status
		}
		Alert alert {
			TimeRange active_period {
				start
				end
			}
			EntitySelector informed_entity
			enum Cause {
                UNKNOWN_CAUSE
                OTHER_CAUSE
                TECHNICAL_PROBLEM
                STRIKE
                DEMONSTRATION
                ACCIDENT
                HOLIDAY
                WEATHER
                MAINTENANCE = 9;
                CONSTRUCTION = 10;
                POLICE_ACTIVITY = 11;
                MEDICAL_EMERGENCY = 12;
            }
            Effect {
                NO_SERVICE = 1;
                REDUCED_SERVICE = 2;

                SIGNIFICANT_DELAYS = 3;

                DETOUR = 4;
                ADDITIONAL_SERVICE = 5;
                MODIFIED_SERVICE = 6;
                OTHER_EFFECT = 7;
                UNKNOWN_EFFECT = 8;
                STOP_MOVED = 9;
            }
            TranslatedString url
            TranslatedString header_text
            TranslatedString description_text
		}
		
		// extentions
	}
}
```

## 参考

[MTA Developer Resources](https://api.mta.info/#/HelpDocument)

[GTFS 实时概览  | Realtime Transit  | Google for Developers](https://developers.google.com/transit/gtfs-realtime?hl=zh-cn)

[GTFS 实时协议缓冲区  | Realtime Transit  | Google for Developers](https://developers.google.com/transit/gtfs-realtime/gtfs-realtime-proto?hl=zh-cn)

[【Protobuf】使用Python实现Protobuf数据框架_python protobuf_Maple_66的博客-CSDN博客](https://blog.csdn.net/qq_41682740/article/details/126571153)

[解决python grpcio.protoc生成的pb文件里面没有类和方法定义的问题_python编译protobuf后没有类_做我的code吧的博客-CSDN博客](https://blog.csdn.net/yueguangMaNong/article/details/127502700)

[在python中使用protobuf - 简书 (jianshu.com)](https://www.jianshu.com/p/1aeb8ee87b99/)