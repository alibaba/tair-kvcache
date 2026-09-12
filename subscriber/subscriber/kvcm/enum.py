from enum import StrEnum


class KvcmReportEventType(StrEnum):
    """Wire event types accepted by KVCM ReportEvent."""

    NODE_REGISTER = "EVENT_NODE_REGISTER"
    HEARTBEAT = "EVENT_HEARTBEAT"
    BLOCK_ADD = "EVENT_BLOCK_ADD"
    BLOCK_DELETE = "EVENT_BLOCK_DELETE"
    BLOCK_SNAPSHOT = "EVENT_BLOCK_SNAPSHOT"
    HOST_DOWN = "EVENT_HOST_DOWN"


class KvcmQueryType(StrEnum):
    """Wire query types accepted by KVCM registerInstance."""

    QT_UNSPECIFIED = "QT_UNSPECIFIED"
    QT_BATCH_GET = "QT_BATCH_GET"
    QT_PREFIX_MATCH = "QT_PREFIX_MATCH"
    QT_REVERSE_ROLL_SW_MATCH = "QT_REVERSE_ROLL_SW_MATCH"
    QT_PREFIX_MATCH_WITH_MAMBA = "QT_PREFIX_MATCH_WITH_MAMBA"


class KvcmStorageType(StrEnum):
    """Storage types supported by the authoritative KVCM protobuf schema."""

    ST_UNSPECIFIED = "ST_UNSPECIFIED"
    ST_3FS = "ST_3FS"
    ST_MOONCAKE = "ST_MOONCAKE"
    ST_TAIRMEMPOOL = "ST_TAIRMEMPOOL"
    ST_NFS = "ST_NFS"
    ST_VCNS_3FS = "ST_VCNS_3FS"
    ST_DUMMY = "ST_DUMMY"
    ST_EVENT_REPORT_L1P5 = "ST_EVENT_REPORT_L1P5"
    ST_EVENT_REPORT_L2 = "ST_EVENT_REPORT_L2"
